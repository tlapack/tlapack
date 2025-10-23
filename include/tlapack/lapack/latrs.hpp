/// @file latrs.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlatrs.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LATRS_HH
#define TLAPACK_LATRS_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"
#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/dot.hpp"
#include "tlapack/blas/dotu.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/blas/trsv.hpp"
#include "tlapack/lapack/ladiv.hpp"
#include "tlapack/lapack/latrs_helpers.hpp"

namespace tlapack {

/** LATRS solves one of the triangular systems
 *
 *  A *x = s*b  or  A**T *x = s*b
 *
 *  with scaling to prevent overflow.  Here A is an upper or lower
 *  triangular matrix, A**T denotes the transpose of A, x and b are
 *  n-element vectors, and s is a scaling factor, usually less than
 *  or equal to 1, chosen so that the components of x will be less than
 *  the overflow threshold.  If the unscaled problem will not cause
 *  overflow, the Level 2 BLAS routine DTRSV is called.  If the matrix A
 *  is singular (A(j,j) = 0 for some j), then s is set to 0 and a
 *  non-trivial solution to A*x = 0 is returned.
 *
 * @param[in] uplo
 *            Specifies whether the matrix A is upper or lower triangular.
 *            - Uplo::Upper: A is upper triangular;
 *            - Uplo::Lower: A is lower triangular.
 *
 * @param[in] trans
 *            Specifies the operation applied to A.
 *           - Op::NoTrans:  solve A * x = s*b  (No transpose);
 *           - Op::Trans:   solve A**T * x = s*b  (Transpose);
 *           - Op::ConjTrans: solve A**H * x = s*b  (Conjugate transpose);
 *
 * @param[in] diag
 *            Specifies whether the matrix A is unit triangular.
 *            - Diag::Unit: A is unit triangular;
 *            - Diag::NonUnit: A is non-unit triangular.
 *
 * @param[in] normin
 *            Specifies whether CNORM has been set or not.
 *           - 'Y':  CNORM contains the column norms on entry;
 *           - 'N':  CNORM is not set on entry.  On exit, the norms will
 *                  be computed and stored in CNORM.
 *
 * @param[in] A n-by-n matrix.
 *            The triangular matrix A.  If UPLO = Uplo::Upper, the leading n by
 *            n upper triangular part of the array A contains the upper
 *            triangular matrix, and the strictly lower triangular part of A is
 *            not referenced.  If UPLO = Uplo::Lower, the leading n by n lower
 *            triangular part of the array A contains the lower triangular
 *            matrix, and the strictly upper triangular part of A is not
 *            referenced.  If DIAG = Diag::Unit, the diagonal elements of A are
 *            also not referenced and are assumed to be 1.
 *
 * @param[in,out] x vector of length n.
 *                On entry, the right hand side B of the triangular system.
 *                On exit, x is overwritten by the solution vector x.
 *
 * @param[out] scale scalar.
 *             The scaling factor s for the triangular system
 *                A * x = s*b  or  A**T* x = s*b.
 *             If SCALE = 0, the matrix A is singular or badly scaled, and
 *             the vector x is an exact or approximate solution to A*x = 0.
 *
 * @ingroup auxiliary
 *
 * @details
 *  A rough bound on x is computed; if that is less than overflow, DTRSV
 *  is called, otherwise, specific code is used which checks for possible
 *  overflow or divide-by-zero at every operation.
 *
 *  A columnwise scheme is used for solving A*x = b.  The basic algorithm
 *  if A is lower triangular is
 *
 *       x[1:n] := b[1:n]
 *       for j = 1, ..., n
 *            x(j) := x(j) / A(j,j)
 *            x[j+1:n] := x[j+1:n] - x(j) * A[j+1:n,j]
 *       end
 *
 *  Define bounds on the components of x after j iterations of the loop:
 *     M(j) = bound on x[1:j]
 *     G(j) = bound on x[j+1:n]
 *  Initially, let M(0) = 0 and G(0) = max{x(i), i=1,...,n}.
 *
 *  Then for iteration j+1 we have
 *     M(j+1) <= G(j) / | A(j+1,j+1) |
 *     G(j+1) <= G(j) + M(j+1) * | A[j+2:n,j+1] |
 *            <= G(j) ( 1 + CNORM(j+1) / | A(j+1,j+1) | )
 *
 *  where CNORM(j+1) is greater than or equal to the infinity-norm of
 *  column j+1 of A, not counting the diagonal.  Hence
 *
 *     G(j) <= G(0) product ( 1 + CNORM(i) / | A(i,i) | )
 *                  1<=i<=j
 *  and
 *
 *     |x(j)| <= ( G(0) / |A(j,j)| ) product ( 1 + CNORM(i) / |A(i,i)| )
 *                                   1<=i< j
 *
 *  Since |x(j)| <= M(j), we use the Level 2 BLAS routine DTRSV if the
 *  reciprocal of the largest M(j), j=1,..,n, is larger than
 *  max(underflow, 1/overflow).
 *
 *  The bound on x(j) is also used to determine when a step in the
 *  columnwise method can be performed without fear of overflow.  If
 *  the computed bound is greater than a large constant, x is scaled to
 *  prevent overflow, but if the bound overflows, x is set to 0, x(j) to
 *  1, and scale to 0, and a non-trivial solution to A*x = 0 is found.
 *
 *  Similarly, a row-wise scheme is used to solve A**T*x = b.  The basic
 *  algorithm for A upper triangular is
 *
 *       for j = 1, ..., n
 *            x(j) := ( b(j) - A[1:j-1,j]**T * x[1:j-1] ) / A(j,j)
 *       end
 *
 *  We simultaneously compute two bounds
 *       G(j) = bound on ( b(i) - A[1:i-1,i]**T * x[1:i-1] ), 1<=i<=j
 *       M(j) = bound on x(i), 1<=i<=j
 *
 *  The initial values are G(0) = 0, M(0) = max{b(i), i=1,..,n}, and we
 *  add the constraint G(j) >= G(j-1) and M(j) >= M(j-1) for j >= 1.
 *  Then the bound on x(j) is
 *
 *       M(j) <= M(j-1) * ( 1 + CNORM(j) ) / | A(j,j) |
 *
 *            <= M(0) * product ( ( 1 + CNORM(i) ) / |A(i,i)| )
 *                      1<=i<=j
 *
 *  and we can safely call DTRSV if 1/M(n) and 1/G(n) are both greater
 *  than max(underflow, 1/overflow).
 */
template <TLAPACK_UPLO uplo_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_MATRIX matrixA_t,
          TLAPACK_VECTOR vectorX_t,
          TLAPACK_VECTOR vectorC_t>
int latrs(uplo_t& uplo,
          trans_t& trans,
          diag_t& diag,
          char& normin,  // TODO: make enum for this?
          const matrixA_t& A,
          vectorX_t& x,
          real_type<type_t<matrixA_t>>& scale,
          vectorC_t& cnorm)
{
    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    scale = (real_t)1;
    const idx_t n = nrows(A);

    // Quick return if possible
    if (n == 0) {
        return 0;
    }

    // TODO: check input

    // Determine machine dependent parameters to control overflow.
    const real_t smlnum = safe_min<real_t>() / ulp<real_t>();
    const real_t bignum = (real_t)1 / smlnum;
    const real_t overflow_threshold = safe_max<real_t>();

    //
    // Compute the column norms
    // Even if the column norms are passed in, we need to
    // scale them if necessary.
    //
    real_t tscal;
    int info_cnorm =
        latrs_calc_cnorm(uplo, trans, diag, normin, A, cnorm, tscal);

    if (info_cnorm == 1) {
        // At least one entry of A is not a valid floating-point entry.
        // Pass this case to trsv for NAN and INF propagation.
        scale = (real_t)0;
        trsv(uplo, trans, diag, A, x);
        return 0;
    }

    //
    // Compute a bound on the computed solution vector to see if the
    // Level 2 BLAS routine DTRSV can be used.
    //
    real_t grow;
    real_t xmax;
    if (tscal != (real_t)1) {
        // If we already had to scale the column norms, then we
        // definitely cannot use TRSV, so no need to compute GROW.
        grow = (real_t)0;
        idx_t j = iamax(x);
        xmax = abs1(x[j]);
    }
    else {
        latrs_calc_growth(uplo, trans, diag, A, x, cnorm, grow, xmax);
    }

    // Initialize scaling factor
    scale = (real_t)1;

    // if ((grow * tscal) > smlnum) {
    //     // Use the Level 2 BLAS solve if the reciprocal of the bound on
    //     // elements of X is not too small.
    //     trsv(uplo, trans, diag, A, x);
    //     return 0;
    // }
    //
    // Use a Level 1 BLAS solve, scaling intermediate results.
    //
    latrs_solve_scaled_system(uplo, trans, diag, A, x, scale, cnorm, tscal,
                              xmax);

    // Scale the column norms by 1/TSCAL for return.
    if (tscal != (real_t)1) {
        scal((real_t)1 / tscal, cnorm);
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LATRS_HH
