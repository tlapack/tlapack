#ifndef __TESTBLAS_DEFINES_HH__
#define __TESTBLAS_DEFINES_HH__

#include <complex>
#define TEST_TYPES float, double, std::complex<float>, std::complex<double>
#define TEST_REAL_TYPES float, double

#include <cstdint>
namespace blas {
    using blas_size_t = std::int64_t;
    using blas_int_t  = std::int64_t;
}

#endif // __TESTBLAS_DEFINES_HH__