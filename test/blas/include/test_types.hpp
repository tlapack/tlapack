// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of T-LAPACK.
// T-LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TESTBLAS_TYPES_HH__
#define __TESTBLAS_TYPES_HH__

#include <complex>

//-----------------------------------------------------------------------------
#ifdef USE_MPFR

    #include <mpreal.h>
    #define TEST_REAL_TYPES float, double, long double, mpfr::mpreal
    #define TEST_CPLX_TYPES std::complex<float>, std::complex<double>, std::complex<long double>, std::complex<mpfr::mpreal>

#else

    #define TEST_REAL_TYPES float, double, long double
    #define TEST_CPLX_TYPES std::complex<float>, std::complex<double>, std::complex<long double>

#endif

//-----------------------------------------------------------------------------
#define TEST_TYPES TEST_REAL_TYPES, TEST_CPLX_TYPES

#endif // __TESTBLAS_TYPES_HH__