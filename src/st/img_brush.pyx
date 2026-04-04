import cython
from libc.math cimport cos, sin, floor, ceil, fabs, fmax, fmin

import numpy as np
cimport numpy as np

np.import_array()

from .utils cimport bilinear_interpolate_imp_double, bilinear_interpolate_imp_uint, max_args, min_args, clip
# from .utils import bilinear_interpolate


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void affine_product(double x, double y, double[:, :] matrix, double[:] pos) noexcept nogil:
    pos[0] = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    pos[1] = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]


cdef void place_brush_imp(unsigned int[:, :] dest, double[:, :, :] brush, double[:] pos, double[:] scale, double angle, unsigned int color):


    cdef double cn = cos(angle)
    cdef double sn = sin(angle)

    # cdef double x00, y00, x10, y10, x01, y01, x11, y11
    # x00 = (cn * (0 - brush.shape[1]/2) - 1/scale[1] * sn * (0 - brush.shape[0]/2)) + pos[1]
    # y00 = (1/scale[0] * sn * (0 - brush.shape[0]/2) + cn * (0 - brush.shape[1]/2)) + pos[0]
    # x10 = (cn * (brush.shape[1] - brush.shape[1]/2) - 1/scale[1] * sn * (0 - brush.shape[0]/2)) + pos[1]
    # y10 = (1/scale[0] * sn * (brush.shape[1] - brush.shape[1]/2) + cn * (0 - brush.shape[0]/2)) + pos[0]
    # x01 = (cn * (0 - brush.shape[1]/2) - 1/scale[1] * sn * (brush.shape[0] - brush.shape[0]/2)) + pos[1]
    # y01 = (1/scale[0] * sn * (0 - brush.shape[0]/2) + cn * (brush.shape[0] - brush.shape[0]/2)) + pos[0]
    # x11 = (cn * (brush.shape[1] - brush.shape[1]/2) - 1/scale[1] * sn * (brush.shape[0] - brush.shape[0]/2)) + pos[1]
    # y11 = (1/scale[0] * sn * (brush.shape[1] - brush.shape[1]/2) + cn * (brush.shape[0] - brush.shape[0]/2)) + pos[0]


    cdef double x00, y00, x10, y10, x01, y01, x11, y11
    x00 = 1/scale[1] * (cn * (0 - brush.shape[1]/2) - sn * (0 - brush.shape[0]/2)) + pos[1]
    y00 = 1/scale[0] * (sn * (0 - brush.shape[0]/2) + cn * (0 - brush.shape[1]/2)) + pos[0]
    x10 = 1/scale[1] * (cn * (brush.shape[1] - brush.shape[1]/2) - sn * (0 - brush.shape[0]/2)) + pos[1]
    y10 = 1/scale[0] * (sn * (brush.shape[1] - brush.shape[1]/2) + cn * (0 - brush.shape[0]/2)) + pos[0]
    x01 = 1/scale[1] * (cn * (0 - brush.shape[1]/2) - sn * (brush.shape[0] - brush.shape[0]/2)) + pos[1]
    y01 = 1/scale[0] * (sn * (0 - brush.shape[0]/2) + cn * (brush.shape[0] - brush.shape[0]/2)) + pos[0]
    x11 = 1/scale[1] * (cn * (brush.shape[1] - brush.shape[1]/2) - sn * (brush.shape[0] - brush.shape[0]/2)) + pos[1]
    y11 = 1/scale[0] * (sn * (brush.shape[1] - brush.shape[1]/2) + cn * (brush.shape[0] - brush.shape[0]/2)) + pos[0]


    cdef double[:] xs = np.array([x00, x10, x01, x11], dtype=np.float64)
    cdef double[:] ys = np.array([y00, y10, y01, y11], dtype=np.float64)

    cdef unsigned int x_min = <unsigned int>clip(min_args(xs), 0, dest.shape[1] - 1)
    cdef unsigned int y_min = <unsigned int>clip(min_args(ys), 0, dest.shape[0] - 1)
    cdef unsigned int x_max = <unsigned int>clip(max_args(xs), 0, dest.shape[1] - 1)
    cdef unsigned int y_max = <unsigned int>clip(max_args(ys), 0, dest.shape[0] - 1)

    # dest[<unsigned int>pos[0], <unsigned int>pos[1]] = 0xFF00FF00

    # dest[y_min, x_min] = 0xFFFF00FF
    # dest[y_max, x_min] = 0xFFFF00FF
    # dest[y_max, x_max] = 0xFFFF00FF
    # dest[y_min, x_max] = 0xFFFF00FF

    # dest[int(y00), int(x00)] = 0xFF000000
    # dest[int(y10), int(x10)] = 0xFF000000
    # dest[int(y01), int(x01)] = 0xFF000000
    # dest[int(y11), int(x11)] = 0xFF000000


    cdef unsigned int x_start, y_start, x_end, y_end
    x_start = <unsigned int>clip(x_min, 0, dest.shape[1] - 1)
    y_start = <unsigned int>clip(y_min, 0, dest.shape[0] - 1)
    x_end = <unsigned int>clip(x_max, 0, dest.shape[1] - 1)
    y_end = <unsigned int>clip(y_max, 0, dest.shape[0] - 1)

    cdef unsigned char r, g, b
    r = (color & 0x00FF0000) >> 16
    g = (color & 0x0000FF00) >> 8
    b = (color & 0x000000FF)

    alpha = brush[..., 3:4]
    brush_argb = np.zeros((brush.shape[0], brush.shape[1]), dtype=np.uint32)
    cdef int i, j
    for i in range(brush.shape[0]):
        for j in range(brush.shape[1]):
            brush_argb[i, j] = (
                  clip(<unsigned int>((1.0 - brush[i, j, 3]) * 255), 0, 255) << 24
                | clip(<unsigned int>((1.0 - brush[i, j, 0]) * r), 0, 255) << 16
                | clip(<unsigned int>((1.0 - brush[i, j, 1]) * g), 0, 255) << 8
                | clip(<unsigned int>((1.0 - brush[i, j, 2]) * b), 0, 255)
            )

    cdef unsigned int[:, :] region = dest[y_start:y_end, x_start:x_end]

    cdef double alpha_pos, inv_alpha_pos
    cdef unsigned int brush_argb_pos, region_pos
    cdef unsigned char brush_r, brush_g, brush_b, region_r, region_g, region_b
    cdef double[:] pos_prime = np.empty(2, dtype=np.float64)
    cdef int x, y
    for x in range(region.shape[0]):
        for y in range(region.shape[1]):

            pos_prime[0] = scale[1] * (cn * (x + x_start - pos[1]) - sn * (y + y_start - pos[0])) + brush.shape[1]/2
            pos_prime[1] = scale[0] * (sn * (x + x_start - pos[1]) + cn * (y + y_start - pos[0])) + brush.shape[0]/2
            if pos_prime[0] < 0 or pos_prime[0] >= brush.shape[1] or pos_prime[1] < 0 or pos_prime[1] >= brush.shape[0]:
                continue

            alpha_pos = bilinear_interpolate_imp_double(alpha, pos_prime[0], pos_prime[1])[0]
            inv_alpha_pos = 1.0 - alpha_pos

            brush_argb_pos = bilinear_interpolate_imp_uint(brush_argb, pos_prime[0], pos_prime[1])

            region_pos = region[x, y]

            # alpha_pos = 0.15
            # inv_alpha_pos = 1.0 - alpha_pos

            # brush_r = 0
            # brush_g = 255
            # brush_b = 0

            brush_r = (brush_argb_pos & 0x00FF0000) >> 16
            brush_g = (brush_argb_pos & 0x0000FF00) >> 8
            brush_b = (brush_argb_pos & 0x000000FF)


            region_r = (region_pos & 0x00FF0000) >> 16
            region_g = (region_pos & 0x0000FF00) >> 8
            region_b = (region_pos & 0x000000FF)

            region[x, y] = (
                  clip(<unsigned int>((region_r * inv_alpha_pos) + (brush_r * alpha_pos)), 0, 255) << 16
                | clip(<unsigned int>((region_g * inv_alpha_pos) + (brush_g * alpha_pos)), 0, 255) << 8
                | clip(<unsigned int>((region_b * inv_alpha_pos) + (brush_b * alpha_pos)), 0, 255)
                ) | 0xFF000000


def place_brush(np.ndarray[unsigned int, ndim=2] dest, np.ndarray[double, ndim=3] brush, np.ndarray[double, ndim=1] pos, np.ndarray[double, ndim=1] scale, double angle, unsigned int color):
    """
    Place a brush onto a destination image using an affine transformation.

    Parameters:
    dest (np.ndarray): The destination image (2D array of unsigned int).
    brush (np.ndarray): The brush image (3D array of double, with shape (height, width, 4)).
    pos (np.ndarray): The position of the brush (1D array of double, with shape (2,)).
    scale (np.ndarray): The scale of the brush (1D array of double, with shape (2,)).
    angle (double): The angle of the brush rotation.
    color (unsigned int): The color to apply to the brush (ARGB format).

    Returns:
    None: The function modifies the dest array in place.
    """
    place_brush_imp(dest, brush, pos, scale, angle, color)