import cython
from libc.math cimport sqrt, abs

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef class Segment:
    cdef Segment seg_r, seg_l
    cdef double[:, :] data

    def __cinit__(self, Segment seg_r = None, Segment seg_l = None, double[:, :] data = None):
        self.seg_r = seg_r
        self.seg_l = seg_l
        self.data = data

    cdef set_data(self, double[:, :] data):
        self.data = data

    cpdef bint is_leaf(self):
        return self.seg_r is None and self.seg_l is None

    cdef set_children(self, Segment seg_r, Segment seg_l):
        self.seg_r = seg_r
        self.seg_l = seg_l


    cpdef double[:, :] concatenate_segments(self):

        if self.is_leaf():
            return self.data

        d_r = self.seg_r.concatenate_segments()
        d_l = self.seg_l.concatenate_segments()

        cdef double[:, :] d = np.empty((d_r.shape[0] + d_l.shape[0], d_r.shape[1]), dtype=np.float64)
        d[: d_r.shape[0], :] = d_r
        d[d_r.shape[0]:, :] = d_l
        return d


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef tuple[double, int] max_abs_dist(double[:, :] points, double[:] start, double[:] line_unit) noexcept nogil:
    cdef int n = points.shape[0]
    cdef int i
    cdef int argmax = 0
    cdef double max_val = 0.0
    cdef double cross
    for i in range(n):
        cross = abs((points[i, 0] - start[0]) * line_unit[1] - (points[i, 1] - start[1]) * line_unit[0])
        if cross > max_val:
            max_val = cross
            argmax = i
    return max_val, argmax


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # enable C division semantics for entire function
cdef Segment simplify_tract_imp(Segment tract, double tolerance=1.0) noexcept:
    # Simplify the tract using the Ramer-Douglas-Peucker algorithm

    cdef double[:, :] pts = tract.concatenate_segments()
    if pts.shape[0] < 3:
        return tract

    # Find the point with the maximum distance from the line
    cdef double[:] start, end
    start, end = pts[0], pts[pts.shape[0]-1]
    cdef double[2] line_vec = [end[0] - start[0], end[1] - start[1]]
    cdef double line_len = sqrt(line_vec[0]**2 + line_vec[1]**2)
    if line_len < 1e-6:
        return Segment(data=np.array([start]))

    cdef double[2] line_unit = [line_vec[0] / line_len, line_vec[1] / line_len]
    cdef double max_dist
    cdef int split_idx
    max_dist, split_idx = max_abs_dist(pts, start, line_unit)  # Perpendicular distance

    if max_dist < tolerance:
        return Segment(data=np.array([start, end]))

    # Recursively simplify the segments
    cdef Segment left = simplify_tract_imp(Segment(data=pts[: split_idx + 1]), tolerance)
    cdef Segment right = simplify_tract_imp(Segment(data=pts[split_idx:]), tolerance)
    return Segment(left, right)


def simplify_tract(cnp.ndarray[cnp.float64_t, ndim=2] tract, double tolerance=1.0):
    # Simplify the tract using the Ramer-Douglas-Peucker algorithm

    cdef Segment simplified = simplify_tract_imp(Segment(data=tract), tolerance)
    return np.array(simplified.concatenate_segments())