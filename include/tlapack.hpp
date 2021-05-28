// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_HH__
#define __TLAPACK_HH__

// BLAS

#include "tblas.hpp"

// Optimized LAPACK

#ifdef USE_LAPACKPP_WRAPPERS

    #ifndef LAPACK_UTIL_HH
        #define LAPACK_UTIL_HH // So as not to include utils from lapack++
    #endif

    #include "lapack/config.h"
    #include "lapack/types.hpp"
    #include "lapack/wrappers.hh" // from lapack++

#endif

// =============================================================================
// Template LAPACK

// Auxiliary routines
// ------------------

#include "lapack/lassq.hpp"
#include "lapack/larf.hpp"
#include "lapack/larfg.hpp"
#include "lapack/lapy2.hpp"
#include "lapack/lapy3.hpp"
#include "lapack/ladiv.hpp"

// QR factorization
// ----------------

#include "lapack/geqr2.hpp"

#endif // __TLAPACK_HH__
