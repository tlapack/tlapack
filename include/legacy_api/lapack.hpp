// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_HH__
#define __TLAPACK_LEGACY_HH__

// Optimized LAPACK

#ifdef USE_LAPACKPP_WRAPPERS
    #ifndef LAPACK_UTIL_HH
        #define LAPACK_UTIL_HH // So as not to include utils from lapack++
    #endif
    #include "legacy_api/lapack/config_lapackppwrappers.h"
    #include "legacy_api/lapack/types.hpp"
    #include "lapack/wrappers.hh" // from lapack++
#endif

#include "legacy_api/lapack/types.hpp"
#include "legacy_api/lapack/utils.hpp"

// =============================================================================
// Template LAPACK

// Auxiliary routines
// ------------------

#include "legacy_api/lapack/larf.hpp"
#include "legacy_api/lapack/larfg.hpp"
#include "legacy_api/lapack/larft.hpp"
#include "legacy_api/lapack/larfb.hpp"
#include "lapack/lapy2.hpp"
#include "lapack/lapy3.hpp"
#include "lapack/ladiv.hpp"
#include "legacy_api/lapack/laset.hpp"
#include "legacy_api/lapack/lacpy.hpp"
#include "legacy_api/lapack/lange.hpp"
#include "legacy_api/lapack/lanhe.hpp"
#include "legacy_api/lapack/lansy.hpp"
#include "legacy_api/lapack/larnv.hpp"
#include "legacy_api/lapack/lascl.hpp"
#include "legacy_api/lapack/lassq.hpp"
#include "legacy_api/lapack/lacgv.hpp"

// QR factorization
// ----------------

#include "legacy_api/lapack/geqr2.hpp"
#include "legacy_api/lapack/ung2r.hpp"
#include "legacy_api/lapack/unm2r.hpp"
#include "legacy_api/lapack/unmqr.hpp"
#include "legacy_api/lapack/potrf.hpp"
#include "legacy_api/lapack/potrs.hpp"

#endif // __TLAPACK_LEGACY_HH__