/// @file latrs_helpers.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlatrs.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LATRS_HELPERS_HH
#define TLAPACK_LATRS_HELPERS_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"
#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/dotu.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/blas/trsv.hpp"

namespace tlapack {

template <TLAPACK_UPLO uplo_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_MATRIX matrixA_t,
          TLAPACK_VECTOR vectorC_t>
int latrs_calc_cnorm(uplo_t& uplo,
                     trans_t& trans,
                     diag_t& diag,
                     char& normin,
                     const matrixA_t& A,
                     vectorC_t& cnorm,
                     real_type<type_t<matrixA_t>>& tscal)
{
    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(A);

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
                // Return with INFO = 1 so that LATRS knows to pass this case
                // to trsv for NAN and INF propagation.
                return 1;
            }
        }
    }
    return 0;
}

template <TLAPACK_UPLO uplo_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_MATRIX matrixA_t,
          TLAPACK_VECTOR vectorX_t,
          TLAPACK_VECTOR vectorC_t>
void latrs_calc_growth(uplo_t& uplo,
                       trans_t& trans,
                       diag_t& diag,
                       const matrixA_t& A,
                       vectorX_t& x,
                       vectorC_t& cnorm,
                       real_type<type_t<matrixA_t>>& grow)
{
    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(A);

    // Determine machine dependent parameters to control overflow.
    const real_t smlnum = safe_min<real_t>() / ulp<real_t>();
    const real_t bignum = (real_t)1 / smlnum;
    const real_t overflow_threshold = safe_max<real_t>();

    idx_t j = iamax(x);
    real_t xmax = abs(x[j]);
    real_t xbnd = xmax;

    idx_t jfirst, jlast, jinc;
    if (trans == Op::NoTrans) {
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
    }
    else {
        if (uplo == Uplo::Upper) {
            jfirst = 0;
            jlast = n;
            jinc = 1;
        }
        else {
            jfirst = n - 1;
            jlast = -1;
            jinc = -1;
        }
    }

    if (trans == Op::NoTrans) {
        // Compute the growth in A * x = b.
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

}  // namespace tlapack

#endif  // TLAPACK_LATRS_HELPERS_HH
