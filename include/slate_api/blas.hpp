// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SLATE_BLAS_HH__
#define __SLATE_BLAS_HH__

// Optimized BLAS

#ifdef USE_BLASPP_WRAPPERS
    
    #include "blas/types.hpp"

    #ifndef BLAS_UTIL_HH
        #define BLAS_UTIL_HH // So as not to include utils from BLAS++
    #endif
    #include "blas/wrappers.hh" // from BLAS++

#endif

// Template BLAS

// =============================================================================
// Level 1 BLAS template implementations

#include "blas/asum.hpp"
#include "blas/axpy.hpp"
#include "blas/copy.hpp"
#include "blas/dot.hpp"
#include "blas/dotu.hpp"
#include "blas/iamax.hpp"
#include "blas/nrm2.hpp"
#include "blas/rot.hpp"
#include "blas/rotg.hpp"
#include "blas/rotm.hpp"
#include "blas/rotmg.hpp"
#include "blas/scal.hpp"
#include "blas/swap.hpp"

// =============================================================================
// Level 2 BLAS template implementations

#include "slate_api/blas/gemv.hpp"
#include "blas/ger.hpp"
#include "blas/geru.hpp"
#include "blas/hemv.hpp"
#include "blas/her.hpp"
#include "blas/her2.hpp"
#include "blas/symv.hpp"
#include "blas/syr.hpp"
#include "blas/syr2.hpp"
// #include "blas/spmv.hpp"
// #include "blas/spr.hpp"
// #include "blas/spr2.hpp"
// #include "blas/sbmv.hpp"
#include "slate_api/blas/trmv.hpp"
#include "blas/trsv.hpp"
// #include "blas/tpmv.hpp"
// #include "blas/tbmv.hpp"
// #include "blas/tpsv.hpp"
// #include "blas/tbsv.hpp"

// =============================================================================
// Level 3 BLAS template implementations

#include "slate_api/blas/gemm.hpp"
#include "blas/hemm.hpp"
#include "blas/herk.hpp"
#include "blas/her2k.hpp"
#include "blas/symm.hpp"
#include "blas/syrk.hpp"
#include "blas/syr2k.hpp"
#include "blas/trmm.hpp"
#include "blas/trsm.hpp"

#endif // __SLATE_BLAS_HH__