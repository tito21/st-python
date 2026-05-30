
cdef void bilinear_interpolate_imp_double(double* image, double x, double y, Py_ssize_t n, Py_ssize_t m, Py_ssize_t l, double* result) noexcept nogil

cdef void bilinear_interpolate_imp_char(unsigned char* image, double x, double y, Py_ssize_t n, Py_ssize_t m, Py_ssize_t l, double* result) noexcept nogil

cdef void bilinear_interpolate_imp_uint(unsigned int* image, double x, double y, Py_ssize_t n, Py_ssize_t m, unsigned int* pixel_value) noexcept nogil

cdef fused dtype:
    int
    double
    unsigned int
    unsigned char

cdef dtype min(dtype a, dtype b) noexcept nogil

cdef dtype max(dtype a, dtype b) noexcept nogil

cdef dtype clip(dtype value, dtype min_value, dtype max_value) noexcept nogil

cdef dtype max_args(dtype[:] args) noexcept nogil
cdef dtype min_args(dtype[:] args) noexcept nogil