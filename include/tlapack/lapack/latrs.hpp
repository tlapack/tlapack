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
#include "tlapack/blas/dotu.hpp"
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

    // Initialize scaling factor
    scale = (real_t)1;

    // Determine machine dependent parameters to control overflow.
    const real_t smlnum = safe_min<real_t>() / ulp<real_t>();
    const real_t bignum = (real_t)1 / smlnum;
    const real_t overflow_threshold = safe_max<real_t>();

    if (normin == 'N') {
        // Compute the 1-norm of each column, not including the diagonal
        if (uplo == Uplo::Upper) {
            for (idx_t j = 0; j < n; ++j) {
                cnorm[j] = asum(slice(col(A, j), range((idx_t)0, j)));
            }
        }
        else {
            for (idx_t j = 0; j < n; ++j) {
                cnorm[j] = asum(slice(col(A, j), range((idx_t)j + 1, n)));
            }
        }
    }

    //
    // Scale the column norms by TSCAL if the maximum element in CNORM is
    // greater than BIGNUM.
    //
    idx_t imax = iamax(cnorm);
    real_t tmax = cnorm[imax];
    real_t tscal;
    if (tmax <= bignum) {
        tscal = (real_t)1;
    }
    else {
        // Avoid NaN generation if entries in CNORM exceed the
        // overflow threshold
        if (tmax <= overflow_threshold) {
            // Case 1: All entries in CNORM are valid floating-point numbers
            tscal = (real_t)1 / (smlnum * tmax);
            scal(tscal, cnorm);
        }
        else {
            // Case 2: At least one column norm of A cannot be represented
            // as floating-point number. Find the offdiagonal entry A( I, J )
            // with the largest absolute value. If this entry is not +/-
            // Infinity, use this value as TSCAL.
            tmax = (real_t)0;
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
                tscal = (real_t)1 / (smlnum * tmax);
                for (idx_t j = 0; j < n; ++j) {
                    if (cnorm[j] <= overflow_threshold) {
                        cnorm[j] = cnorm[j] * tscal;
                    }
                    else {
                        // Recompute the 1-norm without introducing Infinity
                        // in the summation.
                        real_t sum = (real_t)0;
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
                tscal = (real_t)1;
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
    real_t xmax = abs(x[j]);
    real_t xbnd = xmax;

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

    real_t grow;
    if (trans == Op::NoTrans) {
        // Compute the growth in A * x = b.
        if (tscal != (real_t)1) {
            grow = (real_t)0;
        }
        else {
            if (diag == Diag::NonUnit) {
                // A is non-unit triangular.
                //
                // Compute GROW = 1/G(j) and XBND = 1/M(j).
                // Initially, G(0) = max{x(i), i=0,...,n-1}.

                grow = (real_t)1 / max(xbnd, smlnum);
                xbnd = grow;
                for (idx_t j = jfirst; j != jlast; j += jinc) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) {
                        // TODO: check if this is equivalent to goto 50
                        break;
                    }
                    // M(j) = G(j-1) / abs(A(j,j))
                    real_t tjj = abs(A(j, j));
                    xbnd = min(xbnd, min((real_t)1, tjj) * grow);
                    if (tjj + cnorm[j] >= smlnum) {
                        // G(j) = G(j-1) * ( 1 + CNORM(j) / abs(A(j,j)) )
                        grow = grow * (tjj / (tjj + cnorm[j]));
                    }
                    else {
                        // G(j) could overflow, set GROW to 0.
                        grow = (real_t)0;
                    }
                }
                grow = xbnd;
            }
            else {
                // A is unit triangular.
                //
                // Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
                grow = min((real_t)1, (real_t)1 / max(xbnd, smlnum));
                for (idx_t j = jfirst; j != jlast; j += jinc) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) {
                        break;
                    }
                    // G(j) = G(j-1) * ( 1 + CNORM(j) )
                    grow = grow / ((real_t)1 + cnorm[j]);
                }
            }
        }
    }
    else {
        // Compute the growth in A**T * x = b.
        if (tscal != (real_t)1) {
            grow = (real_t)0;
        }
        else {
            if (diag == Diag::NonUnit) {
                // A is non-unit triangular.
                //
                // Compute GROW = 1/G(j) and XBND = 1/M(j).
                // Initially, M(0) = max{x(i), i=0,...,n-1}.

                grow = (real_t)1 / max(xbnd, smlnum);
                xbnd = grow;
                for (idx_t j = jfirst; j != jlast; j += jinc) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) {
                        break;
                    }
                    // G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) )
                    real_t xj = (real_t)1 + cnorm[j];
                    grow = min(grow, xbnd / xj);
                    // M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j))
                    real_t tjj = abs(A(j, j));
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
                grow = min((real_t)1, (real_t)1 / max(xbnd, smlnum));
                for (idx_t j = jfirst; j != jlast; j += jinc) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) {
                        break;
                    }
                    // G(j) = G(j-1) * ( 1 + CNORM(j) )
                    real_t xj = (real_t)1 + cnorm[j];
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
            scale = (real_t)(bignum / xmax);
            scal(scale, x);
            xmax = bignum;
        }

        if (trans == Op::NoTrans) {
            // Solve A * x = b
            for (idx_t j = jfirst; j != jlast; j += jinc) {
                // Compute x(j) = b(j) / A(j,j), scaling x if necessary.
                real_t xj = abs(x[j]);
                T tjjs;
                if (diag == Diag::NonUnit) {
                    tjjs = A(j, j) * tscal;
                }
                else {
                    tjjs = tscal;
                    if (tscal == (real_t)1) {
                        // TODO: figure out the goto
                    }
                }
                real_t tjj = abs(tjjs);
                if (tjj > smlnum) {
                    // abs(A(j,j)) > SMLNUM:
                    if (tjj < (real_t)1) {
                        if (xj > tjj * bignum) {
                            // scale x by 1/b(j)
                            real_t rec = (real_t)1 / xj;
                            scal(rec, x);
                            scale = scale * rec;
                            xmax = xmax * rec;
                        }
                    }
                    x[j] = x[j] / tjjs;
                    xj = abs(x[j]);
                }
                else if (tjj > (real_t)0) {
                    // 0 < abs(A(j,j)) <= SMLNUM:
                    if (xj > tjj * bignum) {
                        // Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM
                        // to avoid overflow when dividing by A(j,j).
                        real_t rec = (tjj * bignum) / xj;
                        if (cnorm[j] > (real_t)1) {
                            // Scale by 1/CNORM(j) to avoid overflow when
                            // multiplying x(j) times column j.
                            rec = rec / cnorm[j];
                        }
                        scal(rec, x);
                        scale = scale * rec;
                        xmax = xmax * rec;
                    }
                    x[j] = x[j] / tjjs;
                    xj = abs(x[j]);
                }
                else {
                    // A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
                    // scale = 0, and compute a solution to A*x = 0.
                    for (idx_t i = 0; i < n; ++i) {
                        x[i] = (real_t)0;
                    }
                    x[j] = (real_t)1;
                    xj = (real_t)1;
                    scale = (real_t)0;
                    xmax = (real_t)0;
                }

                // Scale x if necessary to avoid overflow when adding
                // a multiple of column j of A.
                if (xj > (real_t)1) {
                    real_t rec = (real_t)1 / xj;
                    if (cnorm[j] > (bignum - xmax) * rec) {
                        // Scale x by 1/(2*abs(x(j)))
                        real_t rec = (real_t)0.5 * rec;
                        scal(rec, x);
                        scale = scale * rec;
                    }
                }
                else if (xj * cnorm[j] > (bignum - xmax)) {
                    // Scale x by 1/2
                    scal((real_t)0.5, x);
                    scale = scale * (real_t)0.5;
                }

                if (uplo == Uplo::Upper) {
                    if (j > 0) {
                        // Compute the update
                        // x(0:j) := x(0:j) - x(j) * A(0:j,j)
                        auto Aslice = slice(col(A, j), range(0, j));
                        auto xslice = slice(x, range(0, j));
                        axpy(-x[j] * tscal, Aslice, xslice);
                        idx_t i = iamax(xslice);
                        xmax = abs(x[i]);
                    }
                }
                else {
                    if (j < n - 1) {
                        // Compute the update
                        // x(j+1:n) := x(j+1:n) - x(j) * A(j+1:n,j)
                        auto Aslice = slice(col(A, j), range(j + 1, n));
                        auto xslice = slice(x, range(j + 1, n));
                        axpy(-x[j] * tscal, Aslice, xslice);
                        idx_t i = iamax(xslice) + j + 1;
                        xmax = max(xmax, abs(x[i]));
                    }
                }
            }
        }
        else {
            // Solve A**T * x = b
            for (idx_t j = jfirst; j != jlast; j += jinc) {
                // Compute x(j) = b(j) - sum A(k,j)*x(k).
                //                       k<>j
                real_t xj = abs(x[j]);
                T uscal = tscal;
                real_t rec = (real_t)1 / max(xmax, (real_t)1);
                if (cnorm[j] > (bignum - xj) * rec) {
                    // If x(j) could overflow, scale x by 1/(2*XMAX).
                    rec = rec * (real_t)0.5;
                    T tjjs;
                    if (diag == Diag::Unit) {
                        tjjs = A(j, j) * tscal;
                    }
                    else {
                        tjjs = tscal;
                    }
                    real_t tjj = abs(tjjs);
                    if (tjj > (real_t)1) {
                        // Divide by A(j,j) when scaling x if A(j,j) > 1.
                        rec = min((real_t)1, rec * tjj);
                        uscal = uscal / tjjs;
                    }
                    if (rec < (real_t)1) {
                        scal(rec, x);
                        scale = scale * rec;
                        xmax = xmax * rec;
                    }
                }
                T sumj = (T)0;
                if (uscal == (T)1) {
                    // If the scaling needed for A in the dot product is 1,
                    // call DOT to perform the dot product.
                    if (uplo == Uplo::Upper) {
                        sumj = dotu(slice(col(A, j), range(0, j)),
                                    slice(x, range(0, j)));
                    }
                    else {
                        sumj = dotu(slice(col(A, j), range(j + 1, n)),
                                    slice(x, range(j + 1, n)));
                    }
                }
                else {
                    // Otherwise, use in-line code for the dot product.
                    if (uplo == Uplo::Upper) {
                        for (idx_t i = 0; i < j - 1; ++i) {
                            sumj += A(i, j) * uscal * x[i];
                        }
                    }
                    else {
                        for (idx_t i = j + 1; i < n; ++i) {
                            sumj += A(i, j) * uscal * x[i];
                        }
                    }
                }

                if (uscal == tscal) {
                    // Compute x(j) := ( x(j) - sumj ) / A(j,j) if 1/A(j,j)
                    // was not used to scale the dotproduct.
                    x[j] = x[j] - sumj;
                    xj = abs(x[j]);
                    T tjjs;
                    if (diag == Diag::NonUnit) {
                        tjjs = A(j, j) * tscal;
                    }
                    else {
                        tjjs = tscal;
                        if (tjjs == (real_t)1) {
                            // TODO: fix goto
                        }
                    }
                    // Compute x(j) = x(j) / A(j,j), scaling if necessary.
                    real_t tjj = abs(tjjs);
                    if (tjj > smlnum) {
                        // abs(A(j,j)) > SMLNUM:
                        if (tjj < (real_t)1) {
                            if (xj > tjj * bignum) {
                                // Scale x by 1/abs(x(j))
                                real_t rec = (real_t)1 / xj;
                                scal(rec, x);
                                scale = scale * rec;
                                xmax = xmax * rec;
                            }
                        }
                        x[j] = x[j] / tjjs;
                    }
                    else if (tjj > (real_t)0) {
                        // 0 < abs(A(j,j)) <= SMLNUM:
                        if (xj > tjj * bignum) {
                            // Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM.
                            real_t rec = (tjj * bignum) / xj;
                            scal(rec, x);
                            scale = scale * rec;
                            xmax = xmax * rec;
                        }
                        x[j] = x[j] / tjjs;
                    }
                    else {
                        // A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
                        // scale = 0, and compute a solution to A*x = 0.
                        for (idx_t i = 0; i < n; ++i) {
                            x[i] = (T)0;
                        }
                        x[j] = (T)1;
                        xj = (real_t)1;
                        scale = (real_t)0;
                        xmax = (real_t)0;
                    }
                }
                else {
                    // Compute x(j) := x(j) / A(j,j)  - sumj if the dot
                    // product has already been divided by 1/A(j,j).
                    T tjjs;
                    if (diag == Diag::NonUnit) {
                        tjjs = A(j, j) * tscal;
                    }
                    else {
                        tjjs = tscal;
                    }
                    x[j] = x[j] / tjjs - sumj;
                }
                xmax = max(xmax, abs(x[j]));
            }
        }
        scale = scale / tscal;
    }

    // Scale the column norms by 1/TSCAL for return.
    if (tscal != (real_t)1) {
        scal((real_t)1 / tscal, cnorm);
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LATRS_HH
