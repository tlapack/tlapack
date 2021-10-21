// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SLATE_LAPACK_HH__
#define __SLATE_LAPACK_HH__

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

// #include "lapack/larf.hpp"
// #include "lapack/larfg.hpp"
// #include "lapack/larft.hpp"
// #include "lapack/larfb.hpp"
// #include "lapack/lapy2.hpp"
// #include "lapack/lapy3.hpp"
// #include "lapack/ladiv.hpp"
// #include "lapack/laset.hpp"
#include "lapack/lacpy.hpp"
// #include "lapack/lange.hpp"
// #include "lapack/lansy.hpp"
// #include "lapack/larnv.hpp"
// #include "lapack/lascl.hpp"

// QR factorization
// ----------------

// #include "lapack/geqr2.hpp"
// #include "lapack/org2r.hpp"
// #include "lapack/orm2r.hpp"
// #include "lapack/unmqr.hpp"
// #include "lapack/potrf2.hpp"

#endif // __SLATE_BLAS_HH__