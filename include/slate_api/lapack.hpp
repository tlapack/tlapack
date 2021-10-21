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
#include "lapack/lansy.hpp"
#include "lapack/larnv.hpp"
#include "lapack/lascl.hpp"

// QR factorization
// ----------------

#include "slate_api/lapack/geqr2.hpp"
#include "slate_api/lapack/org2r.hpp"
#include "slate_api/lapack/orm2r.hpp"
#include "slate_api/lapack/unmqr.hpp"
// #include "lapack/potrf2.hpp"

#endif // __SLATE_BLAS_HH__