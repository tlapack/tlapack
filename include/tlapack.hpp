// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HH
#define TLAPACK_HH

// =============================================================================
// Level 1 BLAS template implementations

#include "tlapack/blas/asum.hpp"
#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/copy.hpp"
#include "tlapack/blas/dot.hpp"
#include "tlapack/blas/dotu.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/nrm2.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"
#include "tlapack/blas/rotm.hpp"
#include "tlapack/blas/rotmg.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/blas/swap.hpp"

// =============================================================================
// Level 2 BLAS template implementations

#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/ger.hpp"
#include "tlapack/blas/geru.hpp"
#include "tlapack/blas/hemv.hpp"
#include "tlapack/blas/her.hpp"
#include "tlapack/blas/her2.hpp"
#include "tlapack/blas/symv.hpp"
#include "tlapack/blas/syr.hpp"
#include "tlapack/blas/syr2.hpp"
// #include "tlapack/blas/spmv.hpp"
// #include "tlapack/blas/spr.hpp"
// #include "tlapack/blas/spr2.hpp"
// #include "tlapack/blas/sbmv.hpp"
#include "tlapack/blas/trmv.hpp"
#include "tlapack/blas/trsv.hpp"
// #include "tlapack/blas/tpmv.hpp"
// #include "tlapack/blas/tbmv.hpp"
// #include "tlapack/blas/tpsv.hpp"
// #include "tlapack/blas/tbsv.hpp"

// =============================================================================
// Level 3 BLAS template implementations

#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/hemm.hpp"
#include "tlapack/blas/her2k.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/symm.hpp"
#include "tlapack/blas/syr2k.hpp"
#include "tlapack/blas/syrk.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/blas/trsm.hpp"

// =============================================================================
// Template LAPACK

#include "tlapack/lapack/trtri_recursive.hpp"

// Auxiliary routines
// ------------------

#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/lapack/ladiv.hpp"
#include "tlapack/lapack/lange.hpp"
#include "tlapack/lapack/lanhe.hpp"
#include "tlapack/lapack/lansy.hpp"
#include "tlapack/lapack/lantr.hpp"
#include "tlapack/lapack/lapy2.hpp"
#include "tlapack/lapack/lapy3.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/larft.hpp"
#include "tlapack/lapack/larnv.hpp"
#include "tlapack/lapack/lascl.hpp"
#include "tlapack/lapack/laset.hpp"
#include "tlapack/lapack/lassq.hpp"
#include "tlapack/lapack/lauum_recursive.hpp"
#include "tlapack/lapack/lu_mult.hpp"
#include "tlapack/lapack/rscl.hpp"
#include "tlapack/lapack/transpose.hpp"

// SVD
// ----------------

#include "tlapack/lapack/gebd2.hpp"

// QR factorization
// ----------------

#include "tlapack/lapack/geqr2.hpp"
#include "tlapack/lapack/ung2r.hpp"
#include "tlapack/lapack/unm2r.hpp"
#include "tlapack/lapack/unmqr.hpp"

// LQ factorization
// ----------------

#include "tlapack/lapack/gelq2.hpp"
#include "tlapack/lapack/gelqf.hpp"
#include "tlapack/lapack/ungl2.hpp"

// Solution of positive definite systems
// ----------------

#include "tlapack/lapack/potrf.hpp"
#include "tlapack/lapack/potrs.hpp"
#include "tlapack/lapack/pttrf.hpp"

// Solution of symmetric systems
// ----------------

// #include "tlapack/lapack/hetrf.hpp"

// Sylver equation routines
// ----------------

#include "tlapack/lapack/lasy2.hpp"

// Nonsymmetric standard eigenvalue routines
// ----------------

#include "tlapack/lapack/aggressive_early_deflation.hpp"
#include "tlapack/lapack/gehd2.hpp"
#include "tlapack/lapack/gehrd.hpp"
#include "tlapack/lapack/lahqr.hpp"
#include "tlapack/lapack/lahqr_eig22.hpp"
#include "tlapack/lapack/lahqr_schur22.hpp"
#include "tlapack/lapack/lahqr_shiftcolumn.hpp"
#include "tlapack/lapack/lahr2.hpp"
#include "tlapack/lapack/move_bulge.hpp"
#include "tlapack/lapack/multishift_qr.hpp"
#include "tlapack/lapack/multishift_qr_sweep.hpp"
#include "tlapack/lapack/schur_move.hpp"
#include "tlapack/lapack/schur_swap.hpp"
#include "tlapack/lapack/unghr.hpp"
#include "tlapack/lapack/unmhr.hpp"

// LU
// ----------------

#include "tlapack/lapack/getrf.hpp"

// UL in place, where L and U are coming from the LU factorization of a matrix
// ----------------
#include "tlapack/lapack/ul_mult.hpp"

// Inverse
// ----------------

#include "tlapack/lapack/getri.hpp"

#endif  // TLAPACK_HH
