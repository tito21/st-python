from libc.math cimport sqrt, fmin, fmax, fabs
from libc.stdlib cimport malloc, free

from cython.view cimport array as cvarray
import cython

cimport numpy as np
import numpy as np

from .bezier import fit_curve
from .utils cimport bilinear_interpolate_imp_double
from .simplify_tract import simplify_tract

ctypedef void(*f_ptr)(double, double[:], double[:], void* data) noexcept nogil
ctypedef int(*event_ptr)(double, double[:], void* data) noexcept nogil

cdef struct OdeData:
    double* orientation
    double* stopping
    double stopping_threshold
    double[2] last_out
    int[2] shape


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # enable C division semantics for entire function
cdef inline double norm(double[:] y) noexcept nogil:
    cdef double result = 0.0
    for i in range(y.shape[0]):
        result += y[i] ** 2
    return sqrt(result)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # enable C division semantics for entire function
cdef inline double norm_inf(double[:] y) noexcept nogil:
    cdef double result = 0.0
    for i in range(y.shape[0]):
        result = fmax(result, fabs(y[i]))
    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # enable C division semantics for entire function
cdef void f(double t, double[:] y, double[:] out, void* data) noexcept nogil:
    cdef OdeData* ode_data = <OdeData*>data
    # cdef double[:, :, ::1] orientation = <double[:ode_data.shape[0], :ode_data.shape[1], :2]>ode_data.orientation
    # y is a 2D vector [x, y]
    # Compute the derivative using the orientation
    cdef double[2] vector
    bilinear_interpolate_imp_double(ode_data.orientation, y[0], y[1], ode_data.shape[0], ode_data.shape[1], 2, vector)
    out[0] = vector[0]
    out[1] = vector[1]
    cdef double dir = out[0] * ode_data.last_out[0] + out[1] * ode_data.last_out[1]
    if dir < 0:
        out[0] = -out[0]
        out[1] = -out[1]
    ode_data.last_out[0] = out[0]
    ode_data.last_out[1] = out[1]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # enable C division semantics for entire function
cdef int stopping_condition(double t, double[:] y, void* data) noexcept nogil:
    cdef OdeData* ode_data = <OdeData*>data
    # cdef cvarray stopping = cvarray(shape=(ode_data.shape[0], ode_data.shape[1], 1), itemsize=sizeof(double), format="<d", allocate_buffer=False)
    # stopping.data = <char *>ode_data.stopping
    # cdef double[:, :, ::1] stopping = <double[:ode_data.shape[0], :ode_data.shape[1], :1]>ode_data.stopping
    # Check if the stopping condition is met
    cdef double[1] stopping_value
    bilinear_interpolate_imp_double(ode_data.stopping, y[0], y[1], ode_data.shape[0], ode_data.shape[1], 1, stopping_value)
    if stopping_value[0] < ode_data.stopping_threshold:
        return 0
    return 1

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # enable C division semantics for entire function
cdef int out_of_bounds_condition(double t, double[:] y, void* data) noexcept nogil:
    cdef OdeData* ode_data = <OdeData*>data
    if y[0] < 0 or y[0] > ode_data.shape[0] or y[1] < 0 or y[1] > ode_data.shape[1]:
        return 0
    return 1


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # enable C division semantics for entire function
cdef void RK23_step(f_ptr f, double t, double[:] y, void* data, double h, double[:] err, double[:] k1, double[:] k2, double[:] k3, double[:] k4, double[:] y_in) noexcept nogil:
    """
    We assume that k1 is already computed as f(t, y, k1) before calling this function. This allows us to reuse the k1 array for the next step, reducing function calls.
    """

    cdef int n = y.shape[0]

    for i in range(n):
        y_in[i] = y[i] + h * k1[i] / 2.0
    f(t + h / 2.0, y_in, k2, data)
    for i in range(n):
        y_in[i] = y[i] + h * 3.0 * k2[i] / 4.0
    f(t + 3.0 * h / 4.0, y_in, k3, data)
    for i in range(n):
        y[i] = y[i] + h * (2.0 * k1[i] + 3.0 * k2[i] + 4.0 * k3[i]) / 9.0 # 2nd order solution
    f(t + h, y, k4, data)
    for i in range(n):
        err[i] = h * (-5 * k1[i] / 72.0 + k2[i] / 12.0 + k3[i] / 9.0 - k4[i] / 8.0) # error estimate


@cython.cpow(True)
@cython.cdivision(True)    # enable C division semantics for entire function
cdef solve_ivp(f_ptr func, double[:] t_span, double[:] y0, event_ptr *events, int num_events, void* data=NULL, double tol=1e-3, double max_step=0.5, double min_step=1e-6):
    cdef int n = y0.shape[0]
    cdef double[:] y = np.empty(n, dtype=np.float64)
    cdef double[:] y_in = np.empty(n, dtype=np.float64)
    cdef double[:] k1 = np.empty(n, dtype=np.float64)
    cdef double[:] k2 = np.empty(n, dtype=np.float64)
    cdef double[:] k3 = np.empty(n, dtype=np.float64)
    cdef double[:] k4 = np.empty(n, dtype=np.float64)

    cdef double t = t_span[0]
    cdef double t_end = t_span[1]
    cdef double h = 0.5 * tol ** (1.0/3.0)  # Initial step size guess based on error tolerance
    cdef double[:] err = np.empty(n, dtype=np.float64)
    cdef double err_norm, max_err
    cdef list y_sol = [y0.copy()]
    cdef list t_sol = [t]

    # Initialize k1 with the initial derivative
    cdef int i
    for i in range(n):
        y[i] = y0[i]
    func(t, y, k1, data)

    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        RK23_step(func, t, y, data, h, err, k1, k2, k3, k4, y_in)

        err_norm = norm_inf(err)
        max_err = tol * (1 + norm_inf(y))

        if err_norm < max_err:
            # Accept the step
            y_sol.append(y.copy())
            t_sol.append(t + h)
            k1 = k4

        # Adjust the step size
        h = h * 0.9 * (max_err / err_norm) ** (1.0/3.0) if err_norm > 0 else h
        h = fmax(fmin(h, max_step), min_step)
        t += h
        if num_events > 0:
            for i in range(num_events):
                if not events[i](t, y, data):
                    return (np.asarray(y_sol), np.asarray(t_sol))

    return (np.asarray(y_sol), np.asarray(t_sol))


def compute_tract(double[:, :, ::1] orientation, double[:, ::1] valid_mask, double mask_threshold, tuple starting_point, double max_length, double min_length=1.0, double tolerance=1.0):
    # Compute the tractography streamline
    cdef OdeData data = OdeData(
        orientation = <double*>&orientation[0, 0, 0],
        stopping = <double*>&valid_mask[0, 0],
        stopping_threshold = mask_threshold,
        last_out = [1.0, 0.0],
        shape = [orientation.shape[0], orientation.shape[1]]
    )

    cdef double[:] t_span = np.array([0.0, max_length], dtype=np.float64)
    cdef double[:] y0 = np.array(starting_point, dtype=np.float64)
    cdef event_ptr[2] events = [stopping_condition, out_of_bounds_condition]
    y, t = solve_ivp(f, t_span, y0, data=<void*>&data, events=events, num_events=2, tol=1e-3, max_step=0.1, min_step=1e-6)
    # sol = RK4(ode_system.f, [0, max_length], starting_point, events=events, dt = 1.0)
    tract = simplify_tract(y, tolerance=1.0)
    if len(tract) < 4 or t[-1] < min_length:
        return []
    bezier = fit_curve(tract, error=tolerance * 0.05)
    if np.any(np.isnan(bezier)):
        return []
    return bezier
