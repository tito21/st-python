from math import comb

import numpy as np
from scipy.linalg import lstsq

from .cbezier import fit_cubic

def fit_curve(points, error):
    # points must have shape (2, n)
    # Convert points to a NumPy array
    points = np.array(points)
    # Fit a Bezier curve to the points

    t_hat1 = points[1, :] - points[0, :]
    t_hat1 /= np.linalg.norm(t_hat1) + 1e-6
    t_hat2 = points[-1, :] - points[-2, :]
    t_hat2 /= np.linalg.norm(t_hat2) + 1e-6

    fitted_curve = fit_cubic(points, t_hat1, t_hat2, error)

    return fitted_curve

def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def fit_segment(points, t1_hat, t2_hat, u):
    # points must have shape (2, n)
    # Convert points to a NumPy array
    points = np.array(points)
    num_points = points.shape[0]

    # Construct the Bernstein basis matrix
    # n = 3  # Cubic Bezier
    # B = np.zeros((num_points, n + 1))
    # for i in range(n + 1):
    #     B[:, i] = bernstein_poly(i, n, u)

    # Solve for control points using least squares
    # P = np.linalg.lstsq(B, points, rcond=None)[0]
    # P = lstsq(B, points)[0]
    # bezier_curve = P.reshape((4, 2))

    # Bernstein basis
    B0 = (1 - u)**3
    B1 = 3 * (1 - u)**2 * u
    B2 = 3 * (1 - u) * u**2
    B3 = u**3

    # We want:
    #   P(t) = B0*P0 + B1*(P0 + α u0) + B2*(P3 + β u3) + B3*P3
    #        = [B0*P0 + B1*P0 + B2*P3 + B3*P3] + α B1 u0 + β B2 u3
    #        = base(t) + α B1 u0 + β B2 u3
    base = (
        (B0 + B1)[:, None] * points[0] +
        (B2 + B3)[:, None] * points[-1]
    )

    A_blocks = []
    rhs_blocks = []

    for j in range(2):  # x, y
        A_j = np.column_stack([B1 * t1_hat[j], B2 * t2_hat[j]])  # (N, 2)
        rhs_j = points[:, j] - base[:, j]                   # (N,)
        A_blocks.append(A_j)
        rhs_blocks.append(rhs_j)

    A = np.vstack(A_blocks)        # (2N, 2)
    rhs = np.concatenate(rhs_blocks)  # (2N,)

    alpha_r, alpha_l = lstsq(A, rhs)[0]

    bezier_curve = np.array([
        points[0],
        points[0] + t1_hat * alpha_r,
        points[-1] + t2_hat * alpha_l,
        points[-1]
    ])
    return bezier_curve

def current_error(points, bezier, u):
    num_points = points.shape[0]
    # u = chord_length_parametrization(points)
    error = np.zeros(len(points))
    for i in range(num_points):
        t = u[i]
        B = np.array([bernstein_poly(j, 3, t) for j in range(4)])
        point_on_curve = B @ bezier
        error[i] = np.linalg.norm(point_on_curve - points[i]) ** 2
    error = np.sqrt(error)
    return error

def chord_length_parametrization(points):
    # Compute the chord length for each segment
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    # Compute the cumulative length
    cumulative_length = np.zeros(len(points))
    cumulative_length[1:] = np.cumsum(lengths)
    # Normalize to get parameter values in [0, 1]
    cumulative_length /= cumulative_length[-1] + 1e-6
    return cumulative_length

def nr_root_reparameterize(bezier, points, error):
    num_points = points.shape[0]
    u = chord_length_parametrization(points)

    # Vectorized Bernstein polynomial evaluation
    t = u[:, np.newaxis]
    B = np.array([[bernstein_poly(j, 3, u[i]) for j in range(4)] for i in range(num_points)])

    # Compute derivative of Bernstein basis
    dB = np.zeros((num_points, 4))
    for j in range(4):
        dB[:, j] = 3 * (bernstein_poly(j - 1, 2, u) - bernstein_poly(j, 2, u)) if 0 < j < 4 else (3 * (1 - u) ** 2 if j == 0 else 3 * u ** 2)

    # Vectorized curve and derivative evaluation
    point_on_curve = B @ bezier
    derivative_on_curve = dB @ bezier

    # Vectorized Newton-Raphson update
    numerator = np.sum((point_on_curve - points) * derivative_on_curve, axis=1)
    denominator = np.sum(derivative_on_curve ** 2, axis=1) + 1e-6
    u_prime = u - numerator / denominator

    # Clamp to [0, 1]
    u_prime = np.clip(u_prime, 0.0, 1.0)
    return u_prime

def fit_cubic_lsqr(points, t_hat1, t_hat2, error):

    num_points = points.shape[0]

    if num_points == 2:
        distance = np.linalg.norm(points[1] - points[0]) / 3.0
        bezier = np.array([points[0], points[0] + t_hat1 * distance, points[1] + t_hat2 * distance, points[1]])
        return [bezier]

    # Initial parameterization using chord length
    u = chord_length_parametrization(points)
    bezier_curve = fit_segment(points, t_hat1, t_hat2, u)

    current_error_value = current_error(points, bezier_curve, u)
    mid_index = np.argmax(current_error_value[1:-1]) + 1

    if np.max(current_error_value) < error * 4:
        for _ in range(4):
            u_prime = nr_root_reparameterize(bezier_curve, points, current_error_value)
            bezier_curve = fit_segment(points, t_hat1, t_hat2, u_prime)
            current_error_value = current_error(points, bezier_curve, u_prime)
            if np.max(current_error_value) < error:
                return [bezier_curve]

    t_hat_center = points[mid_index - 1] - points[mid_index + 1]
    t_hat_center /= np.linalg.norm(t_hat_center) + 1e-6
    left_curves = fit_cubic_lsqr(points[:mid_index + 1], t_hat1, t_hat_center, error)
    right_curves = fit_cubic_lsqr(points[mid_index:], -t_hat_center, t_hat2, error)
    return left_curves + right_curves

def fit_curve_lsqr(points, error):
    # points must have shape (2, n)
    # Convert points to a NumPy array
    points = np.array(points)
    num_points = points.shape[0]

    t_hat1 = points[1, :] - points[0, :]
    t_hat1 /= np.linalg.norm(t_hat1) + 1e-6
    t_hat2 = points[-1, :] - points[-2, :]
    t_hat2 /= np.linalg.norm(t_hat2) + 1e-6

    return fit_cubic_lsqr(points, t_hat1, t_hat2, error)

def bezier_point(degree, bezier, t):
    # Compute the point on the Bezier curve at parameter t
    t = t[:, np.newaxis]
    if degree == 0:
        return bezier[0]
    if degree == 1:
        return (1.0 - t) * bezier[0] + t * bezier[1]
    if degree == 2:
        return (1.0 - t)**2 * bezier[0] + 2 * (1.0 - t) * t * bezier[1] + t**2 * bezier[2]
    if degree == 3:
        return (1.0 - t)**3 * bezier[0] + 3 * (1.0 - t)**2 * t * bezier[1] + 3 * (1.0 - t) * t**2 * bezier[2] + t**3 * bezier[3]

    # for i in range(1, degree + 1):
    #     for j in range(0, degree + 1 - i):
    #         Vtemp[j] = (1.0 - t) * Vtemp[j] + t * Vtemp[j + 1]

    # return Vtemp[0]
