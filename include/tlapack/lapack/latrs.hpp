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
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/blas/trsv.hpp"

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
          TLAPACK_VECTOR vectorS_t,
          TLAPACK_VECTOR vectorC_t>
int latrs(uplo_t& uplo,
          trans_t& trans,
          diag_t& diag,
          char& normin,  // TODO: make enum for this?
          int isign,
          const matrixA_t& A,
          vectorX_t& x,
          type_t<matrixA_t>& scale,
          vectorC_t& cnorm)
{
    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;

    scale = (T)1;
    const idx_t n = nrows(A);

    // Quick return if possible
    if (n == 0) {
        return 0;
    }

    // TODO: check input

    const idx_t nrhs = ncols(x);

    // Initialize scaling factors
    for (idx_t j = 0; j < nrhs; ++j) {
        scale = (T)1;
    }

    // Determine machine dependent parameters to control overflow.
    const T smlnum = safe_min<T>() / epsilon<T>();
    const T bignum = (T)1 / smlnum;
    const T overflow_threshold = safe_max<T>();

    if (normin == 'N') {
        // Compute the 1-norm of each column, not including the diagonal
        if (uplo == Uplo::Upper) {
            for (idx_t j = 0; j < n; ++j) {
                cnorm[j] = asum(slice(A.col(j), 0, j - 1));
            }
        }
        else {
            for (idx_t j = 0; j < n; ++j) {
                cnorm[j] = asum(slice(A.col(j), j + 1, n));
            }
        }
    }

    //
    // Scale the column norms by TSCAL if the maximum element in CNORM is
    // greater than BIGNUM.
    //
    idx_t imax = iamax(cnorm);
    T tmax = cnorm[imax];
    T tscal;
    if (tmax <= bignum) {
        tscal = (T)1;
    }
    else {
        // Avoid NaN generation if entries in CNORM exceed the
        // overflow threshold
        if (tmax <= overflow_threshold) {
            // Case 1: All entries in CNORM are valid floating-point numbers
            tscal = one / (smlnum * tmax);
            scal(tscal, cnorm);
        }
        else {
            // Case 2: At least one column norm of A cannot be represented
            // as floating-point number. Find the offdiagonal entry A( I, J )
            // with the largest absolute value. If this entry is not +/-
            // Infinity, use this value as TSCAL.
            tmax = (T)0;
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < j; ++i) {
                        tmax = max(tmax, abs(A(i, j)));
                    }
                }
            }
            else {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j + 1; i < n; ++i) {
                        tmax = max(tmax, abs(A(i, j)));
                    }
                }
            }

            if (tmax <= overflow_threshold) {
                tscal = one / (smlnum * tmax);
                for (idx_t j = 0; j < n; ++j) {
                    if (cnorm[j] <= overflow_threshold) {
                        cnorm[j] = cnorm[j] * tscal;
                    }
                    else {
                        // Recompute the 1-norm without introducing Infinity
                        // in the summation.
                        T sum = (T)0;
                        if (uplo == Uplo::Upper) {
                            for (idx_t i = 0; i < j; ++i) {
                                sum += abs(A(i, j) * tscal);
                            }
                        }
                        else {
                            for (idx_t i = j + 1; i < n; ++i) {
                                sum += abs(A(i, j) * tscal);
                            }
                        }
                        cnorm[j] = sum;
                    }
                }
            }
            else {
                // At least one entry of A is not a valid floating-point entry.
                // Rely on TRSV to propagate Inf and NaN.
                tscal = (T)1;
                trsv(uplo, trans, diag, A, x);
                return 0;
            }
        }
    }

    //
    // Compute a bound on the computed solution vector to see if the
    // Level 2 BLAS routine DTRSV can be used.
    //
    idx_t j = iamax(x);
    T xmax = abs(x[j]);
    T xbnd = xmax;

    idx_t jfirst, jlast, jinc;
    if (uplo == Uplo::Upper) {
        jfirst = n - 1;
        jlast = -1;
        jinc = -1;
    }
    else {
        jfirst = 0;
        jlast = n;
        jinc = 1;
    }

    T grow;
    if (trans == Op::NoTrans) {
        // Compute the growth in A * x = b.
        if (tscal != (T)1) {
            grow = (T)0;
        }
        else {
            if (diag == Diag::NonUnit) {
                // A is non-unit triangular.
                //
                // Compute GROW = 1/G(j) and XBND = 1/M(j).
                // Initially, G(0) = max{x(i), i=0,...,n-1}.

                grow = (T)1 / max(xbnd, smlnum);
                xbnd = grow;
                for (idx_t j = jfirst; j != jlast; j += jinc) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) {
                        // TODO: check if this is equivalent to goto 50
                        break;
                    }
                    // M(j) = G(j-1) / abs(A(j,j))
                    T tjj = abs(A(j, j));
                    xbnd = min(xbnd, min(one, tjj) * grow);
                    if (tjj + cnorm[j] >= smlnum) {
                        // G(j) = G(j-1) * ( 1 + CNORM(j) / abs(A(j,j)) )
                        grow = grow * (tjj / (tjj + cnorm[j]));
                    }
                    else {
                        // G(j) could overflow, set GROW to 0.
                        grow = (T)0;
                    }
                }
                grow = xbnd;
            }
            else {
                // A is unit triangular.
                //
                // Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
                grow = min(one, one / max(xbnd, smlnum));
                for (idx_t j = jfirst; j != jlast; j += jinc) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) {
                        break;
                    }
                    // G(j) = G(j-1) * ( 1 + CNORM(j) )
                    grow = grow / (one + cnorm[j]);
                }
            }
        }
    }
    else {
        // Compute the growth in A**T * x = b.
        if (tscal != (T)1) {
            grow = (T)0;
        }
        else {
            if (diag == Diag::NonUnit) {
                // A is non-unit triangular.
                //
                // Compute GROW = 1/G(j) and XBND = 1/M(j).
                // Initially, M(0) = max{x(i), i=0,...,n-1}.

                grow = (T)1 / max(xbnd, smlnum);
                xbnd = grow;
                for (idx_t j = jfirst; j != jlast; j += jinc) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) {
                        break;
                    }
                    // G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) )
                    T xj = (T)1 + cnorm[j];
                    grow = min(grow, xbnd / xj);
                    // M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j))
                    T tjj = abs(A(j, j));
                    if (xj > tjj) {
                        xbnd = xbnd * (tjj / xj);
                    }
                }
                grow = min(grow, xbnd);
            }
            else {
                // A is unit triangular.
                //
                // Compute GROW = 1/G(j), where G(0) = max{x(i), i=0,...,n-1}.
                grow = min((T)1, (T)1 / max(xbnd, smlnum));
                for (idx_t j = jfirst; j != jlast; j += jinc) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) {
                        break;
                    }
                    // G(j) = G(j-1) * ( 1 + CNORM(j) )
                    T xj = (T)1 + cnorm[j];
                    grow = grow / xj;
                }
            }
        }
    }

    if ((grow * tscal) > smlnum) {
        // Use the Level 2 BLAS solve if the reciprocal of the bound on
        // elements of X is not too small.
        trsv(uplo, trans, diag, A, x);
        return 0;
    }
    else {
        //
        // Use a Level 1 BLAS solve, scaling intermediate results.
        //
        if (xmax > bignum) {
            // Scale X so that its components are less than or equal to
            // BIGNUM in absolute value.
            scale = (T)(bignum / xmax);
            scal(scale, x);
            xmax = bignum;
        }

        if (trans == Op::NoTrans) {
            // Solve A * x = b
            for (idx_t j = jfirst; j != jlast; j += jinc) {
                // Compute x(j) = b(j) / A(j,j), scaling x if necessary.
                T xj = abs(x[j]);
                T tjjs;
                if (diag == Diag::NonUnit) {
                    tjjs = a(j, j) * tscal;
                }
                else {
                    tjjs = tscal;
                    if (tscal == (T)1) {
                        // TODO: check if this is equivalent to goto 100
                        // skip_division; goto skip_division;
                    }
                }
                T tjj = abs(tjjs);
                if (tjj > smlnum) {
                    // abs(A(j,j)) > SMLNUM:
                    if (tjj < (T)1) {
                        if (xj > tjj * bignum) {
                            // scale x by 1/b(j)
                            T rec = (T)1 / xj;
                            scal(rec, x);
                            scale = scale * rec;
                            xmax = xmax * rec;
                        }
                    }
                    x[j] = x[j] / tjjs;
                    xj = abs(x[j]);
                }
            }
        }

        return 0;
    }

}  // namespace tlapack

#endif  // TLAPACK_LATRS_HH
