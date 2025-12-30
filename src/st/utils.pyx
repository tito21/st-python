cimport cython
# cimport numpy as cnp
import numpy as np

from libc.math cimport floor

cdef double fmax(double a, double b) noexcept nogil:
    if a > b:
        return a
    else:
        return b

cdef double fmin(double a, double b) noexcept nogil:
    if a < b:
        return a
    else:
        return b

cdef int min(int a, int b) noexcept nogil:
    if a < b:
        return a
    else:
        return b

cdef double clip(double value, double min_value, double max_value) noexcept nogil:
    return fmax(fmin(value, max_value), min_value)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:] bilinear_interpolate_imp_double(double[:, :, :] image, double x, double y) noexcept:
    cdef int x0 = int(clip(floor(x), 0, image.shape[0] - 1))
    cdef int x1 = min(x0 + 1, image.shape[0] - 1)
    cdef int y0 = int(clip(floor(y), 0, image.shape[1] - 1))
    cdef int y1 = min(y0 + 1, image.shape[1] - 1)
    cdef double dx = x - x0
    cdef double dy = y - y0

    cdef double[:] result = np.empty(image.shape[2], dtype=np.float64)
    cdef int i
    for i in range(image.shape[2]):
        result[i] = ((1 - dx) * (1 - dy) * image[x0, y0, i] +
                      dx * (1 - dy) * image[x1, y0, i] +
                      (1 - dx) * dy * image[x0, y1, i] +
                      dx * dy * image[x1, y1, i])
    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:] bilinear_interpolate_imp_int(unsigned char[:, :, :] image, double x, double y) noexcept:
    cdef int x0 = int(clip(floor(x), 0, image.shape[0] - 1))
    cdef int x1 = min(x0 + 1, image.shape[0] - 1)
    cdef int y0 = int(clip(floor(y), 0, image.shape[1] - 1))
    cdef int y1 = min(y0 + 1, image.shape[1] - 1)
    cdef double dx = x - x0
    cdef double dy = y - y0

    cdef double[:] result = np.empty(image.shape[2], dtype=np.float64)
    cdef int i
    for i in range(image.shape[2]):
        result[i] = ((1 - dx) * (1 - dy) * image[x0, y0, i] +
                      dx * (1 - dy) * image[x1, y0, i] +
                      (1 - dx) * dy * image[x0, y1, i] +
                      dx * dy * image[x1, y1, i])
    return result


def bilinear_interpolate(image, point):
    cdef double x = point[0]
    cdef double y = point[1]
    squeeze = False
    if image.ndim == 2:
        image = image[:, :, None]  # Ensure float type for computation
        squeeze = True

    if image.dtype == np.uint8:
        result = bilinear_interpolate_imp_int(image, x, y)
    elif image.dtype == np.float64:
        result = bilinear_interpolate_imp_double(image, x, y)
    else:
        raise ValueError("Unsupported image data type. Only uint8 and float64 are supported.")
    if squeeze:
        return np.array(result)[0]
    else:
        return np.array(result)