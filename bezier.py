import numpy as np

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

def fit_cubic(points, t_hat1, t_hat2, error):

    num_points = points.shape[0]
    iteration_error = error * 4
    max_iter = 4

    # Use heuristic if region has only two points
    if num_points == 2:

        distance = np.linalg.norm(points[1] - points[0]) / 3.0

        bezier = np.array([points[0], points[0] + t_hat1 * distance, points[1] + t_hat2 * distance, points[1]])
        return [bezier]


    u = chord_length_parametrization(points)
    bezier = generate_bezier(points, u, t_hat1, t_hat2)

    max_error, split_point = compute_max_error(bezier, points, u)

    if max_error < error:
        return [bezier]

    # If error not too large, try reparameterization and iteration
    if max_error < iteration_error:
        for _ in range(max_iter):
            u_prime = reparameterize(bezier, points, u)

            bezier = generate_bezier(points, u_prime, t_hat1, t_hat2)
            max_error, split_point = compute_max_error(bezier, points, u_prime)
            if max_error < error:
                return [bezier]

            u = u_prime

    # fitting failed split and retry
    left = points[:split_point + 1]
    right = points[split_point:]
    t_hat_center = points[split_point - 1] - points[split_point + 1]
    t_hat_center /= np.linalg.norm(t_hat_center) + 1e-6
    return fit_cubic(left, t_hat1, t_hat_center, error) + fit_cubic(right, -t_hat_center, t_hat2, error)


def reparameterize(bezier, points, u):

    # u_prime = np.zeros_like(u)

    # for i in range(len(points)):
    #     u_prime[i] = newton_raphson_root_find(bezier, points[i], u[i])


    u_prime = newton_raphson_root_find(bezier, points, u)

    return u_prime


def newton_raphson_root_find(bezier, point, u):

    Q_u = bezier_point(3, bezier, u)

    Q1 = 3.0 * (bezier[1:] - bezier[:-1])
    Q2 = 2.0 * (bezier[1:] - bezier[:-1])

    Q1_u = bezier_point(2, Q1, u)
    Q2_u = bezier_point(1, Q2, u)

    numerator = np.sum((Q_u - point) * Q1_u)
    denominator = np.sum(Q1_u * Q1_u + (Q_u - point) * Q2_u)
    
    out = np.where(denominator != 0, u - numerator / denominator, u)
    return out


def compute_max_error(bezier, points, u):

    p = bezier_point(3, bezier, u)

    error = np.linalg.norm(p - points, axis=-1)
    split_point = np.argmax(error[1:-1]) + 1
    max_error = error[split_point]

    # max_error = 0
    # split_point = len(points) // 2

    # for i in range(1, len(points) - 1):
    #     p = bezier_point(3, bezier, u[i])

    #     error = np.linalg.norm(p - points[i])
    #     if error > max_error:
    #         max_error = error
    #         split_point = i

    return max_error, split_point


def generate_bezier(points, u, t_hat1, t_hat2):

    A = np.zeros((len(u), 2, 2))

    A[:, 0, 0] = t_hat1[0] * (3 * u * (1.0 - u )**2)
    A[:, 0, 1] = t_hat1[1] * (3 * u**2 * (1.0 - u))
    A[:, 1, 0] = t_hat2[0] * (3 * u * (1.0 - u )**2)
    A[:, 1, 1] = t_hat2[1] * (3 * u**2 * (1.0 - u))


    C = np.zeros((2, 2))
    X = np.zeros((2,))

    C[0, 0] = np.sum(np.vecdot(A[:, 0], A[:, 0]))
    C[0, 1] = np.sum(np.vecdot(A[:, 0], A[:, 1]))
    C[1, 0] = C[0, 1]
    C[1, 1] = np.sum(np.vecdot(A[:, 1], A[:, 1]))

    u = u[:, np.newaxis]
    tmp = points - (points[0] * (1.0 - u)**3 + points[0] * 3 * u * (1.0 - u)**2 + points[-1] * 3 * u**2 * (1.0 - u) + points[-1] * u**3)

    X[0] = np.sum(np.vecdot(A[:, 0], tmp))
    X[1] = np.sum(np.vecdot(A[:, 1], tmp))


    det_C0_C1 = C[0, 0] * C[1, 1] - C[1, 0] * C[0, 1]
    det_C0_X  = C[0, 0] * X[1] - C[1, 0] * X[0]
    det_X_C1  = X[0] * C[1, 1] - X[1] * C[0, 1]
    
    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1
    # alpha = np.array([alpha_l, alpha_r])
    
    # try:
    #     alpha = np.linalg.solve(C, X)

    # except np.linalg.LinAlgError:
    #     # If the system is singular, use heuristic
    #     alpha = np.array([1e-6, 1e-6])

    # if alpha is negative, use heuristic
    seg_length = np.linalg.norm(points[0] - points[-1])
    if alpha_l < 1e-6 * seg_length or alpha_r < 1e-6 * seg_length:
        bezier = np.array([points[0], points[0] + t_hat1 * seg_length, points[1] + t_hat2 * seg_length, points[1]])
        return bezier

    bezier = np.array([points[0], points[0] + t_hat1 * alpha_l, points[-1] + t_hat2 * alpha_r, points[-1]])

    return bezier


def chord_length_parametrization(points):
    # Compute the chord length for each segment
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    # Compute the cumulative length
    cumulative_lengths = np.insert(np.cumsum(lengths), 0, 0)
    # Normalize to get parameters in [0, 1]
    return cumulative_lengths / cumulative_lengths[-1]


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
