// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_HH
#define TLAPACK_BLAS_HH

// Optimized BLAS

#ifdef USE_LAPACKPP_WRAPPERS
    #include "tlapack/optimized/wrappers.hpp"
#endif

// Template BLAS

// =============================================================================
// Level 1 BLAS template implementations

#include "tlapack/legacy_api/blas/asum.hpp"
#include "tlapack/legacy_api/blas/axpy.hpp"
#include "tlapack/legacy_api/blas/copy.hpp"
#include "tlapack/legacy_api/blas/dot.hpp"
#include "tlapack/legacy_api/blas/dotu.hpp"
#include "tlapack/legacy_api/blas/iamax.hpp"
#include "tlapack/legacy_api/blas/nrm2.hpp"
#include "tlapack/legacy_api/blas/rot.hpp"
#include "tlapack/legacy_api/blas/rotg.hpp"
#include "tlapack/legacy_api/blas/rotm.hpp"
#include "tlapack/legacy_api/blas/rotmg.hpp"
#include "tlapack/legacy_api/blas/scal.hpp"
#include "tlapack/legacy_api/blas/swap.hpp"

// =============================================================================
// Level 2 BLAS template implementations

#include "tlapack/legacy_api/blas/gemv.hpp"
#include "tlapack/legacy_api/blas/ger.hpp"
#include "tlapack/legacy_api/blas/geru.hpp"
#include "tlapack/legacy_api/blas/hemv.hpp"
#include "tlapack/legacy_api/blas/her.hpp"
#include "tlapack/legacy_api/blas/her2.hpp"
#include "tlapack/legacy_api/blas/symv.hpp"
#include "tlapack/legacy_api/blas/syr.hpp"
#include "tlapack/legacy_api/blas/syr2.hpp"
// #include "tlapack/blas/spmv.hpp"
// #include "tlapack/blas/spr.hpp"
// #include "tlapack/blas/spr2.hpp"
// #include "tlapack/blas/sbmv.hpp"
#include "tlapack/legacy_api/blas/trmv.hpp"
#include "tlapack/legacy_api/blas/trsv.hpp"
// #include "tlapack/blas/tpmv.hpp"
// #include "tlapack/blas/tbmv.hpp"
// #include "tlapack/blas/tpsv.hpp"
// #include "tlapack/blas/tbsv.hpp"

// =============================================================================
// Level 3 BLAS template implementations

#include "tlapack/legacy_api/blas/gemm.hpp"
#include "tlapack/legacy_api/blas/hemm.hpp"
#include "tlapack/legacy_api/blas/herk.hpp"
#include "tlapack/legacy_api/blas/her2k.hpp"
#include "tlapack/legacy_api/blas/symm.hpp"
#include "tlapack/legacy_api/blas/syrk.hpp"
#include "tlapack/legacy_api/blas/syr2k.hpp"
#include "tlapack/legacy_api/blas/trmm.hpp"
#include "tlapack/legacy_api/blas/trsm.hpp"

#endif // TLAPACK_BLAS_HH