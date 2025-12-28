import numpy as np
cimport numpy as cnp

cnp.import_array()



cdef cnp.ndarray[cnp.float64_t, ndim=2] simplify_tract_imp(cnp.ndarray[cnp.float64_t, ndim=2] tract, double tolerance=1.0):
    # Simplify the tract using the Ramer-Douglas-Peucker algorithm

    if len(tract) < 3:
        return tract

    # Find the point with the maximum distance from the line
    cdef cnp.ndarray[cnp.float64_t, ndim=1] start, end
    start, end = tract[0], tract[-1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] line_vec = end - start
    cdef double line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return np.array([start])

    cdef cnp.ndarray[cnp.float64_t, ndim=1] line_unit = line_vec / line_len
    diff = tract - start
    cdef cnp.ndarray[cnp.float64_t, ndim=1] cross = diff[:, 0] * line_unit[1] - diff[:, 1] * line_unit[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] distances = np.abs(cross)  # Perpendicular distance
    cdef double max_dist = np.max(distances)

    if max_dist < tolerance:
        return np.vstack([start, end])

    # Recursively simplify the segments
    cdef int split_idx = np.argmax(distances)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] left = simplify_tract_imp(tract[: split_idx + 1], tolerance)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] right = simplify_tract_imp(tract[split_idx:], tolerance)
    return np.vstack([left[:-1], right])


def simplify_tract(cnp.ndarray[cnp.float64_t, ndim=2] tract, double tolerance=1.0):
    # Simplify the tract using the Ramer-Douglas-Peucker algorithm

    cdef cnp.ndarray[cnp.float64_t, ndim=2] simplified = simplify_tract_imp(tract, tolerance)
    return np.array(simplified)