// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SLATE_LAPACK_HH__
#define __SLATE_LAPACK_HH__

#include <cstdint> // Defines std::int64_t
#include <cstddef> // Defines std::size_t
#include "slate_api/blas/mdspan.hpp"  // Loads mdspan utilities for the wrappers
#include "plugins/tlapack_mdspan.hpp" // Loads mdspan plugin

// -----------------------------------------------------------------------------
// Integer types BLAS_SIZE_T and BLAS_INT_T

#if defined(USE_BLASPP_WRAPPERS) || defined(USE_LAPACKPP_WRAPPERS)
    #ifndef BLAS_SIZE_T
        #define BLAS_SIZE_T std::int64_t
    #endif
#else
    #ifndef BLAS_SIZE_T
        #define BLAS_SIZE_T std::size_t
    #endif
#endif

#ifndef BLAS_INT_T
    #define BLAS_INT_T std::int64_t
#endif
// -----------------------------------------------------------------------------

namespace lapack {
    using idx_t = BLAS_SIZE_T;
    using int_t = BLAS_INT_T;
}

// Optimized LAPACK

#ifdef USE_LAPACKPP_WRAPPERS

    #ifndef LAPACK_UTIL_HH
        #define LAPACK_UTIL_HH // So as not to include utils from lapack++
    #endif

    #include "slate_api/lapack/config_lapackppwrappers.h"
    #include "lapack/types.hpp"
    #include "lapack/wrappers.hh" // from lapack++

#endif

// =============================================================================
// Template LAPACK

// Auxiliary routines
// ------------------

#include "slate_api/lapack/larf.hpp"
#include "slate_api/lapack/larfg.hpp"
#include "slate_api/lapack/larft.hpp"
#include "slate_api/lapack/larfb.hpp"
#include "lapack/lapy2.hpp"
#include "lapack/lapy3.hpp"
#include "lapack/ladiv.hpp"
#include "slate_api/lapack/laset.hpp"
#include "slate_api/lapack/lacpy.hpp"
#include "slate_api/lapack/lange.hpp"
#include "slate_api/lapack/lansy.hpp"
#include "lapack/larnv.hpp"
#include "slate_api/lapack/lascl.hpp"
#include "slate_api/lapack/lassq.hpp"

// QR factorization
// ----------------

#include "slate_api/lapack/geqr2.hpp"
#include "slate_api/lapack/org2r.hpp"
#include "slate_api/lapack/orm2r.hpp"
#include "slate_api/lapack/unmqr.hpp"
// #include "lapack/potrf2.hpp"

#endif // __SLATE_BLAS_HH__