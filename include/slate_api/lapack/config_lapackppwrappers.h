// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SLATE_CONFIG_WRAPPERS_H__
#define __SLATE_CONFIG_WRAPPERS_H__

#include <stdint.h>
#include "blas/defines.h"
#include "lapack/defines.h"

#ifndef lapack_logical
    #if defined(BLAS_ILP64) || defined(LAPACK_ILP64)
        #define lapack_logical int64_t
    #else
        #define lapack_logical int
    #endif
#endif

// =============================================================================
// Callback logical functions of one, two, or three arguments are used
// to select eigenvalues to sort to the top left of the Schur form in gees and gges.
// The value is selected if function returns TRUE (non-zero).

typedef lapack_logical (*lapack_s_select2) ( float const* omega_real, float const* omega_imag );
typedef lapack_logical (*lapack_s_select3) ( float const* alpha_real, float const* alpha_imag, float const* beta );

typedef lapack_logical (*lapack_d_select2) ( double const* omega_real, double const* omega_imag );
typedef lapack_logical (*lapack_d_select3) ( double const* alpha_real, double const* alpha_imag, double const* beta );

typedef lapack_logical (*lapack_c_select1) ( std::complex<float> const* omega );
typedef lapack_logical (*lapack_c_select2) ( std::complex<float> const* alpha, std::complex<float> const* beta );

typedef lapack_logical (*lapack_z_select1) ( std::complex<double> const* omega );
typedef lapack_logical (*lapack_z_select2) ( std::complex<double> const* alpha, std::complex<double> const* beta );

#endif // __SLATE_CONFIG_WRAPPERS_H__
