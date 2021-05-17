#ifndef __TESTBLAS_TYPES_HH__
#define __TESTBLAS_TYPES_HH__

#include <complex>

//-----------------------------------------------------------------------------
#if defined(USE_GNU_MPFR) && !defined(USE_BLASPP_TEMPLATES)

    #include "mpreal.h"
    #define TEST_REAL_TYPES float, double, long double, mpfr::mpreal
    #define TEST_CPLX_TYPES std::complex<float>, std::complex<double>, std::complex<long double>, std::complex<mpfr::mpreal>

#else

    #define TEST_REAL_TYPES float, double, long double
    #define TEST_CPLX_TYPES std::complex<float>, std::complex<double>, std::complex<long double>

#endif

//-----------------------------------------------------------------------------
#define TEST_TYPES TEST_REAL_TYPES, TEST_CPLX_TYPES

#endif // __TESTBLAS_TYPES_HH__