cimport cython
# cimport numpy as cnp
import numpy as np

from libc.math cimport floor


cdef inline dtype min(dtype a, dtype b) noexcept nogil:
    if a < b:
        return a
    else:
        return b


cdef inline dtype max(dtype a, dtype b) noexcept nogil:
    if a > b:
        return a
    else:
        return b


cdef inline dtype clip(dtype value, dtype min_value, dtype max_value) noexcept nogil:
    return max(min(value, max_value), min_value)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline dtype max_args(dtype[:] args) noexcept nogil:
    cdef dtype result = args[0]
    cdef int i
    for i in range(1, len(args)):
        result = max(result, args[i])
    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline dtype min_args(dtype[:] args) noexcept nogil:
    cdef dtype result = args[0]
    cdef int i
    for i in range(1, len(args)):
        result = min(result, args[i])
    return result


cdef inline Py_ssize_t ravel_index(Py_ssize_t i, Py_ssize_t j, Py_ssize_t k, Py_ssize_t n, Py_ssize_t m, Py_ssize_t l) noexcept nogil:

    cdef Py_ssize_t index
    index = i * (m * l) + j * l + k
    return index



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void bilinear_interpolate_imp_double(double* image, double x, double y, Py_ssize_t n, Py_ssize_t m, Py_ssize_t l, double* result) noexcept nogil:
    cdef int x0 = int(clip(floor(x), 0, n - 1))
    cdef int x1 = min(x0 + 1, n - 1)
    cdef int y0 = int(clip(floor(y), 0, m - 1))
    cdef int y1 = min(y0 + 1, m - 1)
    cdef double dx = x - x0
    cdef double dy = y - y0

    cdef int i
    for i in range(l): # Not thread safe to read from the image
        result[i] = ((1 - dx) * (1 - dy) * image[ravel_index(x0, y0, i, n, m, l)] +
                      dx * (1 - dy) * image[ravel_index(x1, y0, i, n, m, l)] +
                      (1 - dx) * dy * image[ravel_index(x0, y1, i, n, m, l)] +
                      dx * dy * image[ravel_index(x1, y1, i, n, m, l)])


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void bilinear_interpolate_imp_char(unsigned char* image, double x, double y, Py_ssize_t n, Py_ssize_t m, Py_ssize_t l, double* result) noexcept nogil:
    cdef int x0 = int(clip(floor(x), 0, n - 1))
    cdef int x1 = min(x0 + 1, n - 1)
    cdef int y0 = int(clip(floor(y), 0, m - 1))
    cdef int y1 = min(y0 + 1, m - 1)
    cdef double dx = x - x0
    cdef double dy = y - y0

    cdef int i
    for i in range(l):
        result[i] = ((1 - dx) * (1 - dy) * image[ravel_index(x0, y0, i, n, m, l)] +
                      dx * (1 - dy) * image[ravel_index(x1, y0, i, n, m, l)] +
                      (1 - dx) * dy * image[ravel_index(x0, y1, i, n, m, l)] +
                      dx * dy * image[ravel_index(x1, y1, i, n, m, l)])

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void bilinear_interpolate_imp_uint(unsigned int* image, double x, double y, Py_ssize_t n, Py_ssize_t m, unsigned int* pixel_value) noexcept nogil:
    cdef int x0 = <int>(clip(floor(x), 0, n - 1))
    cdef int x1 = min(x0 + 1, n - 1)
    cdef int y0 = <int>(clip(floor(y), 0, m - 1))
    cdef int y1 = min(y0 + 1, m - 1)
    cdef double dx = x - x0
    cdef double dy = y - y0

    # cdef double[:] result = np.empty(4, dtype=np.float64)
    cdef double[4] result = [0, 0, 0, 0]
    cdef int i
    cdef unsigned char pixel_11, pixel_12, pixel_21, pixel_22
    for i in range(4):
        pixel_11 = (image[ravel_index(x0, y0, 0, n, m, 1)] >> (8 * (3 - i))) & 0xFF
        pixel_12 = (image[ravel_index(x1, y0, 0, n, m, 1)] >> (8 * (3 - i))) & 0xFF
        pixel_21 = (image[ravel_index(x1, y1, 0, n, m, 1)] >> (8 * (3 - i))) & 0xFF
        pixel_22 = (image[ravel_index(x1, y1, 0, n, m, 1)] >> (8 * (3 - i))) & 0xFF
        result[i] = ((1 - dx) * (1 - dy) * pixel_11 +
                        dx * (1 - dy) * pixel_12 +
                        (1 - dx) * dy * pixel_21 +
                        dx * dy * pixel_22)

    result[0] = clip(result[0], 0, 255)
    result[1] = clip(result[1], 0, 255)
    result[2] = clip(result[2], 0, 255)
    result[3] = clip(result[3], 0, 255)
    pixel_value[0] = (<unsigned int>(result[0]) << 24) | (<unsigned int>(result[1]) << 16) | (<unsigned int>(result[2]) << 8) | (<unsigned int>(result[3]))


def bilinear_interpolate(image, point):
    cdef double x = point[0]
    cdef double y = point[1]
    squeeze = False
    if image.ndim == 2:
        image = image[:, :, None]  # Add a channel dimension for uniform processing
        squeeze = True

    cdef Py_ssize_t[3] shape = [image.shape[0], image.shape[1], image.shape[2]]

    cdef double[:] result_value = np.empty(image.shape[2], dtype=np.float64)
    cdef unsigned int pixel_value

    cdef double[:, :, ::1] image_pointer_double
    cdef unsigned char[:, :, ::1] image_pointer_char
    cdef unsigned int[:, ::1] image_pointer_int
    if image.dtype == np.uint8:
        image_pointer_char = np.ascontiguousarray(image, dtype=np.uint8)
        bilinear_interpolate_imp_char(&image_pointer_char[0, 0, 0], x, y, shape[0], shape[1], shape[2], &result_value[0])
        result = result_value
    elif image.dtype == np.float64:
        image_pointer_double = np.ascontiguousarray(image, dtype=np.float64)
        bilinear_interpolate_imp_double(&image_pointer_double[0, 0, 0], x, y, shape[0], shape[1], shape[2], &result_value[0])
        result = result_value
    elif image.dtype == np.uint32:
        image_pointer_int = np.ascontiguousarray(image[:, :, 0], dtype=np.uint32)
        bilinear_interpolate_imp_uint(&image_pointer_int[0, 0], x, y, shape[0], shape[1], &pixel_value)
        return pixel_value
    else:
        raise ValueError("Unsupported image data type. Only uint8, uint32 and float64 are supported.")
    if squeeze:
        return np.array(result)[0]
    else:
        return np.array(result)