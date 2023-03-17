/// @file larts.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/zlatrs.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LATRS_HH
#define TLAPACK_LATRS_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/trsv.hpp"
#include "tlapack/lapack/lange.hpp"

namespace tlapack {

template <class matrix_t, class vector_t, class rvector_t>
int latrs(Uplo uplo,
          Op trans,
          Diag diag,
          bool normin,
          matrix_t& A,
          vector_t& x,
          real_type<type_t<matrix_t>>& scale,
          rvector_t& cnorm)
{
    // data traits
    using T = type_t<vector_t>;
    using idx_t = size_type<vector_t>;
    using pair = pair<idx_t, idx_t>;

    // using
    using real_t = real_type<T>;

    // constants
    const real_t one(1);
    const real_t zero(0);
    const real_t half(0.5);
    const real_t two(2);
    const real_t smlnum = safe_min<real_t>() / uroundoff<real_t>();
    const real_t bignum = one / smlnum;
    const real_t rmax = overflow<real_t>();

    scale = one;
    idx_t n = ncols(A);
    if (n == 0) return 0;

    if (!normin) {
        //
        // Compute the 1-norm of each column, not including the diagonal.
        //
        if (uplo == Uplo::Upper) {
            for (idx_t j = 1; j < n; ++j) {
                auto v = slice(A, pair{0, j}, j);
                cnorm[j] = asum(v);
            }
        }
        else {
            for (idx_t j = 0; j < n - 1; ++j) {
                auto v = slice(A, pair{j + 1, n}, j);
                cnorm[j] = asum(v);
            }
        }
    }
    //
    // Scale the column norms by TSCAL if the maximum element in CNORM is
    // greater than BIGNUM.
    //
    auto imax = iamax(cnorm);
    auto tmax = cnorm[imax];
    real_t tscal;
    if (tmax <= bignum) {
        tscal = one;
    }
    else {
        //
        // Avoid NaN generation if entries in CNORM exceed the overflow
        // threshold
        //
        if (tmax <= rmax) {
            // Case 1: All entries in CNORM are valid floating-point numbers
            tscal = one / (smlnum * tmax);
        }
        else {
            // Case 2: At least one column norm of A cannot be represented
            // as floating-point number. Find the offdiagonal entry A( I, J )
            // with the largest absolute value. If this entry is not +/-
            // Infinity, use this value as TSCAL.
            tmax = zero;
            if (uplo == Uplo::Upper) {
                for (idx_t j = 1; j < n; ++j) {
                    auto v = slice(A, pair{0, j}, j);
                    tmax = std::max(lange(Norm::Max, v), tmax);
                }
            }
            else {
                for (idx_t j = 0; j < n - 1; ++j) {
                    auto v = slice(A, pair{j + 1, n}, j);
                    tmax = std::max(lange(Norm::Max, v), tmax);
                }
            }
        }

        if (tmax <= rmax) {
            tscal = one / (smlnum * tmax);
            for (idx_t j = 0; j < n; ++j) {
                if (cnorm[j] <= rmax) {
                    cnorm[j] = cnorm[j] * tscal;
                }
                else {
                    // Recompute the 1-norm without introducing Infinity
                    // in the summation
                    cnorm[j] = zero;
                    if (uplo == Uplo::Upper) {
                        for (idx_t i = 0; i + 1 < j; ++i) {
                            cnorm[j] = cnorm[j] + tscal * tlapack::abs(A(i, j));
                        }
                    }
                    else {
                        for (idx_t i = j + 1; i < n; ++i) {
                            cnorm[j] = cnorm[j] + tscal * tlapack::abs(A(i, j));
                        }
                    }
                }
            }
        }
        else {
            // At least one entry of A is not a valid floating-point entry.
            // Rely on TRSV to propagate Inf and NaN.
            trsv(uplo, trans, diag, A, x);
            return 0;
        }
    }
    //
    // Compute a bound on the computed solution vector to see if the
    // Level 2 BLAS routine DTRSV can be used.
    //
    auto imax2 = iamax(x);
    real_t xmax = tlapack::abs(x[imax2]);
    real_t xbnd = xmax;
    real_t grow;
    if (trans == Op::NoTrans) {
        //
        // Compute the growth in A * x = b.
        //
        idx_t jfirst, jlast, jinc;

        if (tscal != one) {
            grow = zero;
        }
        else {
            if (diag == Diag::NonUnit) {
                //
                // A is non-unit triangular.
                //
                // Compute GROW = 1/G(j) and XBND = 1/M(j).
                // Initially, G(0) = max{x(i), i=1,...,n}.
                //
                grow = one / std::max(xbnd, smlnum);
                xbnd = grow;
                for (idx_t j2 = 0; j2 < n; ++j2) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) break;
                    idx_t j;
                    if (uplo == Uplo::Upper) {
                        j = n - 1 - j2;
                    }
                    else {
                        j = j2;
                    }
                    //
                    // M(j) = G(j-1) / abs(A(j,j))
                    //
                    auto tjj = tlapack::abs(A(j, j));
                    xbnd = std::min(xbnd, std::min(one, tjj) * grow);
                    if (tjj + cnorm[j] >= smlnum) {
                        // G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) )
                        grow = grow * (tjj / (tjj + cnorm[j]));
                    }
                    else {
                        // G(j) could overflow, set GROW to 0.
                        grow = zero;
                    }
                }
                grow = xbnd;
            }
            else {
                //
                // A is unit triangular.
                //
                // Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
                //
                grow = std::min(one, one / std::max(xbnd, smlnum));
                for (idx_t j2 = 0; j2 < n; ++j2) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) break;
                    idx_t j;
                    if (uplo == Uplo::Upper) {
                        j = n - 1 - j2;
                    }
                    else {
                        j = j2;
                    }
                    // G(j) = G(j-1)*( 1 + CNORM(j) )
                    grow = grow * (one / (one + cnorm[j]));
                }
            }
        }
    }
    else {
        //
        //  Compute the growth in A**T * x = b.
        //
        idx_t jfirst, jlast, jinc;

        if (tscal != one) {
            grow = zero;
        }
        else {
            if (diag == Diag::NonUnit) {
                //
                // A is non-unit triangular.
                //
                // Compute GROW = 1/G(j) and XBND = 1/M(j).
                // Initially, G(0) = max{x(i), i=1,...,n}.
                //
                grow = one / std::max(xbnd, smlnum);
                xbnd = grow;
                for (idx_t j2 = 0; j2 < n; ++j2) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) break;
                    idx_t j;
                    if (uplo == Uplo::Upper) {
                        j = j2;
                    }
                    else {
                        j = n - 1 - j2;
                    }
                    //
                    // G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) )
                    //
                    auto xj = one + cnorm[j];
                    grow = std::min(grow, xbnd / xj);
                    //
                    // M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j))
                    //
                    auto tjj = tlapack::abs(A(j, j));
                    if (xj > tjj) xbnd = xbnd * (tjj / xj);
                }
                grow = std::min(grow, xbnd);
            }
            else {
                //
                // A is unit triangular.
                //
                // Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
                //
                grow = std::min(one, one / std::max(xbnd, smlnum));
                for (idx_t j2 = 0; j2 < n; ++j2) {
                    // Exit the loop if the growth factor is too small.
                    if (grow <= smlnum) break;
                    idx_t j;
                    if (uplo == Uplo::Upper) {
                        j = j2;
                    }
                    else {
                        j = n - 1 - j2;
                    }
                    // G(j) = ( 1 + CNORM(j) )*G(j-1)
                    auto xj = one + cnorm[j];
                    grow = grow / xj;
                }
            }
        }
    }
    //
    // Actually solve the system
    //
    if ((grow * tscal) > smlnum) {
        //
        // Use the Level 2 BLAS solve if the reciprocal of the bound on
        // elements of X is not too small.
        //
        trsv(uplo, trans, diag, A, x);
    }
    else {
        //
        // Use a Level 1 BLAs solve, scaling intermediate results.
        //

        // TODO: write this, false assertion for now
        assert(false);
    }
    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LATRS_HH
