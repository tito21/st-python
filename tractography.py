from collections import namedtuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from bezier import fit_curve


def compute_structural_tensor(image, rho=1.0, sigma=1.0):
    # Compute the gradients
    dx = gaussian_filter(image, sigma=(sigma, sigma), order=(1, 0))
    dy = gaussian_filter(image, sigma=(sigma, sigma), order=(0, 1))

    # Compute the elements of the structure tensor
    J11 = gaussian_filter(dx * dx, sigma=rho)
    J22 = gaussian_filter(dy * dy, sigma=rho)
    J12 = gaussian_filter(dx * dy, sigma=rho)

    return J11, J22, J12


def compute_eigensystem(J11, J22, J12):
    # Compute the eigenvalues and eigenvectors of the structure tensor and return them in ascending order
    # A = np.stack(
    #     [np.stack([J11, J12], axis=-1), np.stack([J12, J22], axis=-1)], axis=-1
    # )
    # eigvals, eigvecs = np.linalg.eigh(A)
    # sorted_indices = np.argsort(eigvals, axis=-1)
    # eigvals_true = np.take_along_axis(eigvals, sorted_indices, axis=-1)
    # eigvecs_true = np.take_along_axis(
    #     eigvecs, sorted_indices[..., np.newaxis, :], axis=-1
    # )
    # cond = eigvecs_true[..., 1, :] > 0
    # eigvecs_true[..., 1, :] = np.where(cond, eigvecs_true[..., 1, :], -eigvecs_true[..., 1, :])
    # eigvecs_true[..., 0, :] = np.where(cond, eigvecs_true[..., 0, :], -eigvecs_true[..., 0, :])

    angle = 0.5 *np.arctan2(2 * J12, J22 - J11)
    c = np.cos(angle)
    s = np.sin(angle)
    v1 = np.stack([c, -s], axis=-1)
    v2 = np.stack([s, c], axis=-1)

    descriminant = np.sqrt((J11 - J22) ** 2 + 4 * J12**2)
    lambda1 = 0.5 * (J11 + J22 + descriminant)
    lambda2 = 0.5 * (J11 + J22 - descriminant)
    eigvals = np.stack([lambda2, lambda1], axis=-1)

    eigvecs = np.stack([v1, v2], axis=-1)
    # cond = eigvecs[..., 1, :] > 0
    # eigvecs[..., 1, :] = np.where(cond, eigvecs[..., 1, :], -eigvecs[..., 1, :])
    # eigvecs[..., 0, :] = np.where(cond, eigvecs[..., 0, :], -eigvecs[..., 0, :])

    # print(
    #     "Eigensystem check:",
    #     np.allclose(eigvals, eigvals_true),
    #     np.allclose(np.abs(np.vecdot(eigvecs[..., 0], eigvecs_true[..., 0])), 1),
    #     np.allclose(np.abs(np.vecdot(eigvecs[..., 1], eigvecs_true[..., 1])), 1),
    #     np.allclose(eigvecs[..., 0], eigvecs_true[..., 0]),
    #     np.allclose(eigvecs[..., 1], eigvecs_true[..., 1]),
    # )

    # return eigvals_true, eigvecs_true
    return eigvals, eigvecs


def compute_gradient_normal_orientation(image, sigma=1.0):
    dx = gaussian_filter(image, sigma=(sigma, sigma), order=(1, 0))
    dy = gaussian_filter(image, sigma=(sigma, sigma), order=(0, 1))

    return np.stack([-dy, dx], axis=-1)
    # return np.stack([dx, dy], axis=-1)


def coherence(eigenvalues):
    # Compute the coherence of the structure tensor
    lambda1, lambda2 = eigenvalues[:, :, 0], eigenvalues[:, :, 1]
    lambda_sum = lambda1 + lambda2
    lambda_diff = lambda1 - lambda2

    coherence = (lambda_diff / lambda_sum) ** 2
    coherence[lambda_sum == 0] = 0

    return coherence


def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def bilinear_interpolate(image, point):
    x, y = point
    x0 = clip(int(np.floor(x)), 0, image.shape[0] - 1)
    x1 = min(x0 + 1, image.shape[0] - 1)
    y0 = clip(int(np.floor(y)), 0, image.shape[1] - 1)
    y1 = min(y0 + 1, image.shape[1] - 1)

    dx = x - x0
    dy = y - y0

    top = (1 - dx) * image[x0, y0] + dx * image[x1, y0]
    bottom = (1 - dx) * image[x0, y1] + dx * image[x1, y1]

    return (1 - dy) * top + dy * bottom


class ODESystem:

    def __init__(self, orientation, stopping, stopping_threshold):
        x, y = np.arange(orientation.shape[0]), np.arange(orientation.shape[1])

        # self.orientation_interpolated = RegularGridInterpolator(
        #     (x, y), orientation, method="linear", bounds_error=False, fill_value=0
        # )

        # self.stopping_interpolated = RegularGridInterpolator(
        #     (x, y), stopping, method="linear", bounds_error=False, fill_value=0
        # )

        self.orientation = orientation
        self.stopping = stopping

        self.shape = orientation.shape
        self.stopping_threshold = stopping_threshold
        self.last_vector = np.array([1.0, 0.0])

        ODESystem.stopping_condition.terminal = True
        ODESystem.out_of_bounds_condition.terminal = True

        self.events = [self.stopping_condition, self.out_of_bounds_condition]

    def f(self, t, y):
        # y is a 2D vector [x, y]
        # Compute the derivative using the orientation
        # vector = self.orientation_interpolated(y)[0]
        vector = bilinear_interpolate(self.orientation, y)
        if np.dot(vector, self.last_vector) < 0:
            vector = -vector
        self.last_vector = vector
        return vector

    def stopping_condition(self, t, y):
        # Check if the stopping condition is met
        if bilinear_interpolate(self.stopping, y) < self.stopping_threshold:
            # if self.stopping_interpolated(y) < self.stopping_threshold:
            return False
        return True

    def out_of_bounds_condition(self, t, y):
        if y[0] < 0 or y[0] > self.shape[0] or y[1] < 0 or y[1] > self.shape[1]:
            return False
        return True


def RK4(f, y0, t_limits, dt=1, events=None):
    # Runge-Kutta 4th order method
    t = t_limits[0]
    t1 = t_limits[1]
    y = y0
    trajectory = [y0]

    while t < t1:
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt / 2 * k1)
        k3 = f(t + dt / 2, y + dt / 2 * k2)
        k4 = f(t + dt, y + dt * k3)

        y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        if events is not None:
            for event in events:
                if not event(t, y):
                    return namedtuple("Trajectory", ["y", "t"])(np.array(trajectory).T, [t])
        trajectory.append(y)
        t += dt

    return namedtuple("Trajectory", ["y", "t"])(np.array(trajectory).T, [t])


def compute_tract(ode_system, starting_point, max_length, min_length=1.0, tolerance=1.0):
    # Compute the tractography streamline

    events = ode_system.events
    # Integrate the ODE system
    sol = solve_ivp(
        ode_system.f, [0, max_length], starting_point, events=events, method="RK23"
    )
    # sol = RK4(ode_system.f, [0, max_length], starting_point, events=events, dt = 1.0)
    tract = simplify_tract([v for v in sol.y.T], tolerance=1.0)
    if len(tract) < 4 or np.linalg.norm(sol.t[-1]) < min_length:
        return []
    bezier = fit_curve(tract, error=tolerance * 0.5)
    if np.any(np.isnan(bezier)):
        return []
    return bezier


def simplify_tract(tract, tolerance=1.0):
    # Simplify the tract using the Ramer-Douglas-Peucker algorithm
    if len(tract) < 3:
        return tract

    # Find the point with the maximum distance from the line
    start, end = tract[0], tract[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return [start]

    line_unit = line_vec / line_len
    distances = np.abs(np.cross(tract - start, line_unit))  # Perpendicular distance
    max_dist = np.max(distances)

    if max_dist < tolerance:
        return [start, end]

    # Recursively simplify the segments
    split_idx = np.argmax(distances)
    left = simplify_tract(tract[: split_idx + 1], tolerance)
    right = simplify_tract(tract[split_idx:], tolerance)
    return left[:-1] + right
