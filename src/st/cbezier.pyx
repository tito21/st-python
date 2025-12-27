import numpy as np
cimport numpy as cnp

cnp.import_array()

def fit_cubic(
    cnp.ndarray[cnp.float64_t, ndim=2] points,
    cnp.ndarray[cnp.float64_t, ndim=1] t_hat1,
    cnp.ndarray[cnp.float64_t, ndim=1] t_hat2,
    double error
):

    cdef int num_points = points.shape[0]
    cdef double iteration_error = error * 4
    cdef int max_iter = 4

    cdef cnp.ndarray[cnp.float64_t, ndim=2] bezier
    # Use heuristic if region has only two points
    if num_points == 2:

        distance = np.linalg.norm(points[1] - points[0]) / 3.0
        bezier = np.array([points[0], points[0] + t_hat1 * distance, points[1] + t_hat2 * distance, points[1]])
        return [bezier]


    cdef cnp.ndarray[cnp.float64_t, ndim=1] u = chord_length_parametrization(points)
    bezier = generate_bezier(points, u, t_hat1, t_hat2)

    max_error, split_point = compute_max_error(bezier, points, u)

    if max_error < error:

        bezier, finished = iterations(bezier, points, u, max_iter, t_hat1, t_hat2, error)
        if finished:
            return [bezier]

    # If error not too large, try reparameterization and iteration
    if max_error < iteration_error:
        bezier, finished = iterations(bezier, points, u, max_iter, t_hat1, t_hat2, error)
        if finished:
            return [bezier]

    # fitting failed split and retry
    cdef cnp.ndarray[cnp.float64_t, ndim=2] left = points[:split_point + 1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] right = points[split_point:]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] t_hat_center = points[split_point - 1] - points[split_point + 1]
    t_hat_center /= np.linalg.norm(t_hat_center) + 1e-6
    return fit_cubic(left, t_hat1, t_hat_center, error) + fit_cubic(right, -t_hat_center, t_hat2, error)


cdef tuple[cnp.ndarray, bint] iterations(
    cnp.ndarray[cnp.float64_t, ndim=2] bezier,
    cnp.ndarray[cnp.float64_t, ndim=2] points,
    cnp.ndarray[cnp.float64_t, ndim=1] u,
    int max_iter, cnp.ndarray[cnp.float64_t, ndim=1]
    t_hat1, cnp.ndarray[cnp.float64_t, ndim=1]
    t_hat2,
    double error
):
    cdef int _
    cdef cnp.ndarray[cnp.float64_t, ndim=1] u_prime
    cdef int split_point
    cdef double max_error
    for _ in range(max_iter):
        u_prime = newton_raphson_root_find(bezier, points, u)

        bezier = generate_bezier(points, u_prime, t_hat1, t_hat2)
        max_error, split_point = compute_max_error(bezier, points, u_prime)
        if max_error < error:
            return bezier, True

        u = u_prime

    return bezier, False


cdef cnp.ndarray[cnp.float64_t, ndim=1] newton_raphson_root_find(
    cnp.ndarray[cnp.float64_t, ndim=2] bezier,
    cnp.ndarray[cnp.float64_t, ndim=2] point,
    cnp.ndarray[cnp.float64_t, ndim=1] u
):

    cdef cnp.ndarray[cnp.float64_t, ndim=2] Q_u = bezier_point_c(3, bezier, u)

    Q1 = 3.0 * (bezier[1:] - bezier[:-1])
    Q2 = 2.0 * (bezier[1:] - bezier[:-1])

    cdef cnp.ndarray[cnp.float64_t, ndim=2] Q1_u = bezier_point_c(2, Q1, u)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Q2_u = bezier_point_c(1, Q2, u)

    numerator = np.sum((Q_u - point) * Q1_u)
    denominator = np.sum(Q1_u * Q1_u + (Q_u - point) * Q2_u)

    out = np.where(denominator != 0, u - numerator / denominator, u)
    return out


cdef tuple[double, int] compute_max_error(
    cnp.ndarray[cnp.float64_t, ndim=2] bezier,
    cnp.ndarray[cnp.float64_t, ndim=2] points,
    cnp.ndarray[cnp.float64_t, ndim=1] u
):

    cdef cnp.ndarray[cnp.float64_t, ndim=2] p = bezier_point_c(3, bezier, u)

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


cdef cnp.ndarray[cnp.float64_t, ndim=2] generate_bezier(
    cnp.ndarray[cnp.float64_t, ndim=2] points,
    cnp.ndarray[cnp.float64_t, ndim=1] u,
    cnp.ndarray[cnp.float64_t, ndim=1] t_hat1,
    cnp.ndarray[cnp.float64_t, ndim=1] t_hat2
):

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

    cdef cnp.ndarray[cnp.float64_t, ndim=2] u_expanded = u[:, np.newaxis]
    tmp = points - (points[0] * (1.0 - u_expanded)**3 + points[0] * 3 * u_expanded * (1.0 - u_expanded)**2 + points[-1] * 3 * u_expanded**2 * (1.0 - u_expanded) + points[-1] * u_expanded**3)

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


cdef cnp.ndarray[cnp.float64_t, ndim=1] chord_length_parametrization(cnp.ndarray[cnp.float64_t, ndim=2] points):
    # Compute the chord length for each segment
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    # Compute the cumulative length
    cumulative_lengths = np.insert(np.cumsum(lengths), 0, 0)
    # Normalize to get parameters in [0, 1]
    return cumulative_lengths / cumulative_lengths[-1]


cdef cnp.ndarray[cnp.float64_t, ndim=2] bezier_point_c(
    int degree, cnp.ndarray[cnp.float64_t, ndim=2] bezier,
    cnp.ndarray[cnp.float64_t, ndim=1] t
):
    # Compute the point on the Bezier curve at parameter t
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_expanded = t[:, np.newaxis]
    if degree == 0:
        return bezier[0]
    elif degree == 1:
        return (1.0 - t_expanded) * bezier[0] + t_expanded * bezier[1]
    elif degree == 2:
        return (1.0 - t_expanded)**2 * bezier[0] + 2 * (1.0 - t_expanded) * t_expanded * bezier[1] + t_expanded**2 * bezier[2]
    elif degree == 3:
        return (1.0 - t_expanded)**3 * bezier[0] + 3 * (1.0 - t_expanded)**2 * t_expanded * bezier[1] + 3 * (1.0 - t_expanded) * t_expanded**2 * bezier[2] + t_expanded**3 * bezier[3]
    else:
        raise ValueError("Unsupported Bezier degree. Only degrees 0 to 3 are supported.")
