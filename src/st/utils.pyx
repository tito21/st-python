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
cdef double[:] bilinear_interpolate_imp_char(unsigned char[:, :, :] image, double x, double y) noexcept:
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
cdef unsigned int bilinear_interpolate_imp_uint(unsigned int[:, :] image, double x, double y) noexcept:
    cdef int x0 = <int>(clip(floor(x), 0, image.shape[0] - 1))
    cdef int x1 = min(x0 + 1, image.shape[0] - 1)
    cdef int y0 = <int>(clip(floor(y), 0, image.shape[1] - 1))
    cdef int y1 = min(y0 + 1, image.shape[1] - 1)
    cdef double dx = x - x0
    cdef double dy = y - y0

    cdef double[:] result = np.empty(4, dtype=np.float64)
    cdef int i
    cdef unsigned char pixel_11, pixel_12, pixel_21, pixel_22
    for i in range(4):
        pixel_11 = (image[x0, y0] >> (8 * (3 - i))) & 0xFF
        pixel_12 = (image[x1, y0] >> (8 * (3 - i))) & 0xFF
        pixel_21 = (image[x0, y1] >> (8 * (3 - i))) & 0xFF
        pixel_22 = (image[x1, y1] >> (8 * (3 - i))) & 0xFF
        result[i] = ((1 - dx) * (1 - dy) * pixel_11 +
                        dx * (1 - dy) * pixel_12 +
                        (1 - dx) * dy * pixel_21 +
                        dx * dy * pixel_22)

    result[0] = clip(result[0], 0, 255)
    result[1] = clip(result[1], 0, 255)
    result[2] = clip(result[2], 0, 255)
    result[3] = clip(result[3], 0, 255)
    cdef unsigned int pixel_value = (<unsigned int>(result[0]) << 24) | (<unsigned int>(result[1]) << 16) | (<unsigned int>(result[2]) << 8) | (<unsigned int>(result[3]))
    return pixel_value



def bilinear_interpolate(image, point):
    cdef double x = point[0]
    cdef double y = point[1]
    squeeze = False
    if image.ndim == 2:
        image = image[:, :, None]  # Add a channel dimension for uniform processing
        squeeze = True

    if image.dtype == np.uint8:
        result = bilinear_interpolate_imp_char(image, x, y)
    elif image.dtype == np.float64:
        result = bilinear_interpolate_imp_double(image, x, y)
    elif image.dtype == np.uint32:
        result = bilinear_interpolate_imp_uint(image[:, :, 0], x, y)
    else:
        raise ValueError("Unsupported image data type. Only uint8, uint32 and float64 are supported.")
    if squeeze:
        return np.array(result)[0]
    else:
        return np.array(result)