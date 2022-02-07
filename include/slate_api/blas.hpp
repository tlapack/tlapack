// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SLATE_BLAS_HH__
#define __SLATE_BLAS_HH__

#include "slate_api/blas/mdspan.hpp"  // Loads mdspan utilities for the wrappers
#include "plugins/tlapack_mdspan.hpp" // Loads mdspan plugin
#include "slate_api/blas/types.hpp"

// Optimized BLAS

#ifdef USE_BLASPP_WRAPPERS
    #ifndef BLAS_UTIL_HH
        #define BLAS_UTIL_HH // So as not to include utils from BLAS++
    #endif
    #include "blas/wrappers.hh" // from BLAS++
#endif

// Template BLAS

// =============================================================================
// Level 1 BLAS template implementations

#include "slate_api/blas/asum.hpp"
#include "slate_api/blas/axpy.hpp"
#include "slate_api/blas/copy.hpp"
#include "slate_api/blas/dot.hpp"
#include "slate_api/blas/dotu.hpp"
#include "slate_api/blas/iamax.hpp"
#include "slate_api/blas/nrm2.hpp"
#include "slate_api/blas/rot.hpp"
#include "slate_api/blas/rotg.hpp"
#include "slate_api/blas/rotm.hpp"
#include "slate_api/blas/rotmg.hpp"
#include "slate_api/blas/scal.hpp"
#include "slate_api/blas/swap.hpp"

// =============================================================================
// Level 2 BLAS template implementations

#include "slate_api/blas/gemv.hpp"
#include "slate_api/blas/ger.hpp"
#include "slate_api/blas/geru.hpp"
#include "slate_api/blas/hemv.hpp"
#include "slate_api/blas/her.hpp"
#include "slate_api/blas/her2.hpp"
#include "slate_api/blas/symv.hpp"
#include "slate_api/blas/syr.hpp"
#include "slate_api/blas/syr2.hpp"
// #include "blas/spmv.hpp"
// #include "blas/spr.hpp"
// #include "blas/spr2.hpp"
// #include "blas/sbmv.hpp"
#include "slate_api/blas/trmv.hpp"
#include "slate_api/blas/trsv.hpp"
// #include "blas/tpmv.hpp"
// #include "blas/tbmv.hpp"
// #include "blas/tpsv.hpp"
// #include "blas/tbsv.hpp"

// =============================================================================
// Level 3 BLAS template implementations

#include "slate_api/blas/gemm.hpp"
#include "slate_api/blas/hemm.hpp"
#include "slate_api/blas/herk.hpp"
#include "slate_api/blas/her2k.hpp"
#include "slate_api/blas/symm.hpp"
#include "slate_api/blas/syrk.hpp"
#include "slate_api/blas/syr2k.hpp"
#include "slate_api/blas/trmm.hpp"
#include "slate_api/blas/trsm.hpp"

#endif // __SLATE_BLAS_HH__