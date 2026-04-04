
import numpy as np
cimport numpy as cnp
import cython

from libc.math cimport sqrt

cnp.import_array()



def fit_cubic(
    cnp.ndarray[cnp.float64_t, ndim=2] points,
    cnp.ndarray[cnp.float64_t, ndim=1] t_hat1,
    cnp.ndarray[cnp.float64_t, ndim=1] t_hat2,
    double error
):
    cdef double[2] t_hat1_c = t_hat1
    cdef double[2] t_hat2_c = t_hat2
    return np.array(fit_cubic_imp(points, t_hat1_c, t_hat2_c, error))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@ cython.cdivision(True)    # enable C division semantics for entire function
cdef list[double[4][2]] fit_cubic_imp(
    double[:, :] points,
    double[2] t_hat1,
    double[2] t_hat2,
    double error
) noexcept:

    cdef int num_points = points.shape[0]
    cdef double iteration_error = error * 4
    cdef int max_iter = 4

    cdef double[4][2] bezier
    # Use heuristic if region has only two points
    if num_points == 2:
        distance = distance_point(points[1], points[0]) / 3.0
        get_bezier_control_points(
            points[0],
            points[1],
            t_hat1,
            t_hat2,
            distance,
            distance,
            bezier
        )
        return [bezier]


    cdef cnp.ndarray[cnp.float64_t, ndim=1] u = np.array(chord_length_parametrization(points))
    generate_bezier(points, u, t_hat1, t_hat2, bezier)

    cdef int split_point
    cdef double max_error
    max_error, split_point = compute_max_error(bezier, points, u)

    # If error not too large, try reparameterization and iteration
    if max_error < iteration_error:
        for _ in range(max_iter):
            newton_raphson_root_find(bezier, points, u)  # updates u in place
            generate_bezier(points, u, t_hat1, t_hat2, bezier)
            max_error, split_point = compute_max_error(bezier, points, u)
            if max_error < error:
                return [bezier]

    # fitting failed split and retry
    left = points[:split_point + 1]
    right = points[split_point:]
    cdef double[2] t_hat_center = [points[split_point - 1, 0] - points[split_point + 1, 0], points[split_point - 1, 1] - points[split_point + 1, 1]]
    cdef double norm = sqrt(t_hat_center[0] ** 2 + t_hat_center[1] ** 2) + 1e-6
    t_hat_center[0] /= norm
    t_hat_center[1] /= norm
    cdef double[2] t_hat_center_inverse = [-t_hat_center[0], -t_hat_center[1]]
    return fit_cubic_imp(left, t_hat1, t_hat_center, error) + \
           fit_cubic_imp(right, t_hat_center_inverse, t_hat2, error)



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@ cython.cdivision(True)    # enable C division semantics for entire function
cdef void newton_raphson_root_find(
    double[4][2] bezier,
    double[:, :] point,
    cnp.ndarray[cnp.float64_t, ndim=1] u
) noexcept:

    cdef cnp.ndarray[cnp.float64_t, ndim=2] Q_u = np.array(bezier_point_c(3, bezier, u))

    cdef double[3][2] Q1
    Q1[0][0] = 3.0 * (bezier[1][0] - bezier[0][0])
    Q1[0][1] = 3.0 * (bezier[1][1] - bezier[0][1])
    Q1[1][0] = 3.0 * (bezier[2][0] - bezier[1][0])
    Q1[1][1] = 3.0 * (bezier[2][1] - bezier[1][1])
    Q1[2][0] = 3.0 * (bezier[3][0] - bezier[2][0])
    Q1[2][1] = 3.0 * (bezier[3][1] - bezier[2][1])
    cdef double[3][2] Q2
    Q2[0][0] = 2.0 * (bezier[1][0] - bezier[0][0])
    Q2[0][1] = 2.0 * (bezier[1][1] - bezier[0][1])
    Q2[1][0] = 2.0 * (bezier[2][0] - bezier[1][0])
    Q2[1][1] = 2.0 * (bezier[2][1] - bezier[1][1])
    Q2[2][0] = 2.0 * (bezier[3][0] - bezier[2][0])
    Q2[2][1] = 2.0 * (bezier[3][1] - bezier[2][1])

    cdef cnp.ndarray[cnp.float64_t, ndim=2] Q1_u = np.array(bezier_point_c(2, Q1, u))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Q2_u = np.array(bezier_point_c(1, Q2, u))

    numerator = np.sum((Q_u - point) * Q1_u)
    denominator = np.sum(Q1_u * Q1_u + (Q_u - point) * Q2_u)

    u = np.where(denominator != 0, u - numerator / denominator, u)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@ cython.cdivision(True)    # enable C division semantics for entire function
cdef tuple[double, int] compute_max_error(
    double[4][2] bezier,
    double[:, :] points,
    cnp.ndarray[cnp.float64_t, ndim=1] u
) noexcept:

    cdef double[:, :] p = bezier_point_c(3, bezier, u)

    cdef int i
    cdef int split_point = 0
    cdef double max_error = 0.0
    cdef double error
    for i in range(1, points.shape[0] - 1):
        error = sqrt((p[i, 0] - points[i, 0])**2 + (p[i, 1] - points[i, 1])**2)
        if error > max_error:
            max_error = error
            split_point = i

    return max_error, split_point


@cython.cdivision(True)    # enable C division semantics for entire function
cdef void generate_bezier(
    double[:, :] points,
    cnp.ndarray[cnp.float64_t, ndim=1] u,
    double[2] t_hat1,
    double[2] t_hat2,
    double[4][2] bezier
) noexcept:

    A = np.zeros((len(u), 2, 2))

    A[:, 0, 0] = t_hat1[0] * (3 * u * (1.0 - u )**2)
    A[:, 0, 1] = t_hat1[1] * (3 * u**2 * (1.0 - u))
    A[:, 1, 0] = t_hat2[0] * (3 * u * (1.0 - u )**2)
    A[:, 1, 1] = t_hat2[1] * (3 * u**2 * (1.0 - u))


    cdef double[2][2] C = np.zeros((2, 2))
    cdef double[2] X = np.zeros((2,))

    # C[0, 0] = np.sum(np.vecdot(A[:, 0], A[:, 0]))
    # C[0, 1] = np.sum(np.vecdot(A[:, 0], A[:, 1]))
    # C[1, 0] = C[0, 1]
    # C[1, 1] = np.sum(np.vecdot(A[:, 1], A[:, 1]))

    C[0][0] = sum_vecdot(A[:, 0], A[:, 0])
    C[0][1] = sum_vecdot(A[:, 0], A[:, 1])
    C[1][0] = C[0][1]
    C[1][1] = sum_vecdot(A[:, 1], A[:, 1])

    cdef cnp.ndarray[cnp.float64_t, ndim=2] u_expanded = u[:, np.newaxis]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] points_np = np.array(points)
    tmp = points_np - (points_np[0] * (1.0 - u_expanded)**3 + points_np[0] * 3 * u_expanded * (1.0 - u_expanded)**2 + points_np[-1] * 3 * u_expanded**2 * (1.0 - u_expanded) + points_np[-1] * u_expanded**3)

    X[0] = sum_vecdot(A[:, 0], tmp)
    X[1] = sum_vecdot(A[:, 1], tmp)


    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X  = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1  = X[0] * C[1][1] - X[1] * C[0][1]

    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

    # try:
    #     alpha = np.linalg.solve(C, X)

    # except np.linalg.LinAlgError:
    #     # If the system is singular, use heuristic
    #     alpha = np.array([1e-6, 1e-6])
    # alpha_l = alpha[0]
    # alpha_r = alpha[1]

    # Bernstein basis
    # B0 = (1 - u)**3
    # B1 = 3 * (1 - u)**2 * u
    # B2 = 3 * (1 - u) * u**2
    # B3 = u**3

    # # We want:
    # #   P(t) = B0*P0 + B1*(P0 + α u0) + B2*(P3 + β u3) + B3*P3
    # #        = [B0*P0 + B1*P0 + B2*P3 + B3*P3] + α B1 u0 + β B2 u3
    # #        = base(t) + α B1 u0 + β B2 u3
    # base = (
    #     (B0 + B1)[:, None] * points[0] +
    #     (B2 + B3)[:, None] * points[-1]
    # )

    # A_blocks = []
    # rhs_blocks = []

    # for j in range(2):  # x, y
    #     A_j = np.column_stack([B1 * t_hat1[j], B2 * t_hat2[j]])  # (N, 2)
    #     rhs_j = points[:, j] - base[:, j]                   # (N,)
    #     A_blocks.append(A_j)
    #     rhs_blocks.append(rhs_j)

    # A = np.vstack(A_blocks)        # (2N, 2)
    # rhs = np.concatenate(rhs_blocks)  # (2N,)

    # alpha_r, alpha_l = np.linalg.lstsq(A, rhs, rcond=None)[0]


    # if alpha is negative, use heuristic

    cdef double[:] start, end
    start = points[0]
    end = points[points.shape[0]-1]
    seg_length = distance_point(start, end)
    if alpha_l < 1e-6 * seg_length or alpha_r < 1e-6 * seg_length:
        get_bezier_control_points(
            start,
            end,
            t_hat1,
            t_hat2,
            seg_length,
            seg_length,
            bezier
        )
        return

    get_bezier_control_points(
        start,
        end,
        t_hat1,
        t_hat2,
        alpha_l,
        alpha_r,
        bezier
    )

    return


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@ cython.cdivision(True)    # enable C division semantics for entire function
cdef double[:] chord_length_parametrization(double[:, :] points) noexcept:
    # Compute the chord length for each segment
    cdef double[:] lengths = distance_array_array(points[1:], points[:points.shape[0]-1])
    # Compute the cumulative length
    cdef double[:] cumulative_lengths = np.zeros(points.shape[0], dtype=np.float64)

    cdef int i
    for i in range(1, points.shape[0]):
        cumulative_lengths[i] = cumulative_lengths[i-1] + lengths[i-1]
    # Normalize to get parameters in [0, 1]
    for i in range(points.shape[0]):
        cumulative_lengths[i] /= cumulative_lengths[points.shape[0]-1] + 1e-10
    return cumulative_lengths


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:, :] bezier_point_c(
    int degree,
    double[4][2] bezier,
    double[:] t
) noexcept:
    # Compute the point on the Bezier curve at parameter t

    cdef double[:, :] out = np.empty((t.shape[0], 2), dtype=np.float64)
    cdef int i
    if degree == 0:
        for i in range(t.shape[0]):
            out[i, 0] = bezier[0][0]
            out[i, 1] = bezier[0][1]
    elif degree == 1:
        for i in range(t.shape[0]):
            out[i, 0] = (1.0 - t[i]) * bezier[0][0] + t[i] * bezier[1][0]
            out[i, 1] = (1.0 - t[i]) * bezier[0][1] + t[i] * bezier[1][1]
    elif degree == 2:
        for i in range(t.shape[0]):
            out[i, 0] = (1.0 - t[i])**2 * bezier[0][0] + 2 * (1.0 - t[i]) * t[i] * bezier[1][0] + t[i]**2 * bezier[2][0]
            out[i, 1] = (1.0 - t[i])**2 * bezier[0][1] + 2 * (1.0 - t[i]) * t[i] * bezier[1][1] + t[i]**2 * bezier[2][1]
    elif degree == 3:
        for i in range(t.shape[0]):
            out[i, 0] = (1.0 - t[i])**3 * bezier[0][0] + 3 * (1.0 - t[i])**2 * t[i] * bezier[1][0] + 3 * (1.0 - t[i]) * t[i]**2 * bezier[2][0] + t[i]**3 * bezier[3][0]
            out[i, 1] = (1.0 - t[i])**3 * bezier[0][1] + 3 * (1.0 - t[i])**2 * t[i] * bezier[1][1] + 3 * (1.0 - t[i]) * t[i]**2 * bezier[2][1] + t[i]**3 * bezier[3][1]
    else:
        return np.zeros((len(t), 2))
    return out


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void get_bezier_control_points(
    double[:] p_l,
    double[:] p_r,
    double[2] t_hat1,
    double[2] t_hat2,
    double alpha_l,
    double alpha_r,
    double[4][2] bezier
) noexcept nogil:

    cdef int i
    for i in range(2):
        bezier[0][i] = p_l[i]
        bezier[1][i] = p_l[i] + t_hat1[i] * alpha_l
        bezier[2][i] = p_r[i] + t_hat2[i] * alpha_r
        bezier[3][i] = p_r[i]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double distance_point(double[:] p1, double[:] p2) noexcept nogil:
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:] distance_array_array(double[:, :] points1, double[:, :] points2) noexcept:
    cdef int n = points1.shape[0]
    cdef double[:] dists = np.empty(n, dtype=np.float64)
    cdef int i
    for i in range(n):
        dists[i] = sqrt((points1[i, 0] - points2[i, 0])**2 + (points1[i, 1] - points2[i, 1])**2)
    return dists


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double sum_vecdot(double[:, :] a, double[:, :] b) noexcept nogil:
    cdef int n = a.shape[0]
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += a[i, 0] * b[i, 0] + a[i, 1] * b[i, 1]
    return total