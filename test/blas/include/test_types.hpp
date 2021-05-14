#ifndef __TESTBLAS_TYPES_HH__
#define __TESTBLAS_TYPES_HH__

//-----------------------------------------------------------------------------
#define TEST_REAL_TYPES float, double
#define TEST_CPLX_TYPES std::complex<float>, std::complex<double>

// //-----------------------------------------------------------------------------
// #include "mpreal.h"
// #include <complex>
// #define TEST_REAL_TYPES mpfr::mpreal
// #define TEST_CPLX_TYPES std::complex<mpfr::mpreal>

//-----------------------------------------------------------------------------
#define TEST_TYPES TEST_REAL_TYPES, TEST_CPLX_TYPES

#endif // __TESTBLAS_TYPES_HH__