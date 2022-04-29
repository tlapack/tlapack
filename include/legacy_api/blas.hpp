// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_BLAS_HH__
#define __TLAPACK_BLAS_HH__

// Template BLAS

// =============================================================================
// Level 1 BLAS template implementations

#include "legacy_api/blas/asum.hpp"
#include "legacy_api/blas/axpy.hpp"
#include "legacy_api/blas/copy.hpp"
#include "legacy_api/blas/dot.hpp"
#include "legacy_api/blas/dotu.hpp"
#include "legacy_api/blas/iamax.hpp"
#include "legacy_api/blas/nrm2.hpp"
#include "legacy_api/blas/rot.hpp"
#include "legacy_api/blas/rotg.hpp"
#include "legacy_api/blas/rotm.hpp"
#include "legacy_api/blas/rotmg.hpp"
#include "legacy_api/blas/scal.hpp"
#include "legacy_api/blas/swap.hpp"

// =============================================================================
// Level 2 BLAS template implementations

#include "legacy_api/blas/gemv.hpp"
#include "legacy_api/blas/ger.hpp"
#include "legacy_api/blas/geru.hpp"
#include "legacy_api/blas/hemv.hpp"
#include "legacy_api/blas/her.hpp"
#include "legacy_api/blas/her2.hpp"
#include "legacy_api/blas/symv.hpp"
#include "legacy_api/blas/syr.hpp"
#include "legacy_api/blas/syr2.hpp"
// #include "blas/spmv.hpp"
// #include "blas/spr.hpp"
// #include "blas/spr2.hpp"
// #include "blas/sbmv.hpp"
#include "legacy_api/blas/trmv.hpp"
#include "legacy_api/blas/trsv.hpp"
// #include "blas/tpmv.hpp"
// #include "blas/tbmv.hpp"
// #include "blas/tpsv.hpp"
// #include "blas/tbsv.hpp"

// =============================================================================
// Level 3 BLAS template implementations

#include "legacy_api/blas/gemm.hpp"
#include "legacy_api/blas/hemm.hpp"
#include "legacy_api/blas/herk.hpp"
#include "legacy_api/blas/her2k.hpp"
#include "legacy_api/blas/symm.hpp"
#include "legacy_api/blas/syrk.hpp"
#include "legacy_api/blas/syr2k.hpp"
#include "legacy_api/blas/trmm.hpp"
#include "legacy_api/blas/trsm.hpp"

#endif // __TLAPACK_BLAS_HH__