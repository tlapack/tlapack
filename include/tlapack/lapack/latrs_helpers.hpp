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
int latrs_calc_cnorm(const uplo_t& uplo,
                     const trans_t& trans,
                     const diag_t& diag,
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
            if constexpr (is_complex<T>) {
                tscal = (real_t)0.5 / (smlnum * tmax);
            }
            else {
                tscal = (real_t)1 / (smlnum * tmax);
            }
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
                        tmax = max(tmax,
                                   max(abs(real(A(i, j))), abs(imag(A(i, j)))));
                    }
                }
            }
            else {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j + 1; i < n; ++i) {
                        tmax = max(tmax,
                                   max(abs(real(A(i, j))), abs(imag(A(i, j)))));
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
                        if constexpr (is_complex<T>) {
                            tscal = (real_t)2 * tscal;
                        }
                        real_t sum = (real_t)0;
                        if (uplo == Uplo::Upper) {
                            for (idx_t i = 0; i < j; ++i) {
                                if constexpr (is_complex<T>) {
                                    sum += tscal *
                                           (abs(real(A(i, j)) / ((real_t)2.)) +
                                            abs(imag(A(i, j)) / ((real_t)2.)));
                                }
                                else {
                                    sum += abs(A(i, j) * tscal);
                                }
                            }
                        }
                        else {
                            for (idx_t i = j + 1; i < n; ++i) {
                                if constexpr (is_complex<T>) {
                                    sum += tscal *
                                           (abs(real(A(i, j)) / ((real_t)2.)) +
                                            abs(imag(A(i, j)) / ((real_t)2.)));
                                }
                                else {
                                    sum += abs(A(i, j) * tscal);
                                }
                            }
                        }
                        if constexpr (is_complex<T>) {
                            tscal = (real_t)0.5 * tscal;
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
void latrs_calc_growth(const uplo_t& uplo,
                       const trans_t& trans,
                       const diag_t& diag,
                       const matrixA_t& A,
                       vectorX_t& x,
                       vectorC_t& cnorm,
                       real_type<type_t<matrixA_t>>& grow,
                       real_type<type_t<matrixA_t>>& xmax)
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

    if constexpr (is_complex<T>) {
        xmax = (real_t)0;
        for (idx_t i = 0; i < n; ++i) {
            xmax = max(xmax, abs(real(x[i]) / (real_t)2.) +
                                 abs(imag(x[i]) / (real_t)2.));
        }
    }
    else {
        idx_t j = iamax(x);
        xmax = abs(x[j]);
    }
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

            if constexpr (is_complex<T>) {
                grow = (real_t)0.5 / max(xbnd, smlnum);
            }
            else {
                grow = (real_t)1 / max(xbnd, smlnum);
            }
            xbnd = grow;
            for (idx_t j = jfirst; j != jlast; j += jinc) {
                // Exit the loop if the growth factor is too small.
                if (grow <= smlnum) {
                    // TODO: check if this is equivalent to goto 50
                    break;
                }
                real_t tjj = abs1(A(j, j));
                if (tjj >= smlnum) {
                    // M(j) = G(j-1) / abs(A(j,j))
                    xbnd = min(xbnd, min((real_t)1, tjj) * grow);
                }
                else {
                    // M(j) could overflow, set XBND to 0.
                    xbnd = (real_t)0;
                }

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
            if constexpr (is_complex<T>) {
                grow = min((real_t)1, (real_t)0.5 / max(xbnd, smlnum));
            }
            else {
                grow = min((real_t)1, (real_t)1 / max(xbnd, smlnum));
            }
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
        // Compute the growth in A**T * x = b.
        if (diag == Diag::NonUnit) {
            // A is non-unit triangular.
            //
            // Compute GROW = 1/G(j) and XBND = 1/M(j).
            // Initially, M(0) = max{x(i), i=0,...,n-1}.

            if constexpr (is_complex<T>) {
                grow = (real_t)0.5 / max(xbnd, smlnum);
            }
            else {
                grow = (real_t)1 / max(xbnd, smlnum);
            }
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
                real_t tjj = abs1(A(j, j));
                if (tjj >= smlnum) {
                    if (xj > tjj) {
                        xbnd = xbnd * (tjj / xj);
                    }
                }
                else {
                    // M(j) could overflow, set XBND to 0.
                    xbnd = (real_t)0;
                }
            }
            grow = min(grow, xbnd);
        }
        else {
            // A is unit triangular.
            //
            // Compute GROW = 1/G(j), where G(0) = max{x(i), i=0,...,n-1}.
            if constexpr (is_complex<T>) {
                grow = min((real_t)1, (real_t)0.5 / max(xbnd, smlnum));
            }
            else {
                grow = min((real_t)1, (real_t)1 / max(xbnd, smlnum));
            }
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

template <TLAPACK_UPLO uplo_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_MATRIX matrixA_t,
          TLAPACK_VECTOR vectorX_t,
          TLAPACK_VECTOR vectorC_t>
void latrs_solve_scaled_system(const uplo_t& uplo,
                               const trans_t& trans,
                               const diag_t& diag,
                               const matrixA_t& A,
                               vectorX_t& x,
                               real_type<type_t<matrixA_t>>& scale,
                               vectorC_t& cnorm,
                               const real_type<type_t<matrixA_t>>& tscal,
                               real_type<type_t<matrixA_t>>& xmax)
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

    // Initialize scaling factor
    scale = (real_t)1;

    if (xmax > bignum) {
        // Scale X so that its components are less than or equal to
        // BIGNUM in absolute value.
        scale = (real_t)(bignum / xmax);
        scal(scale, x);
        xmax = bignum;
    }

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
        // Solve A * x = b
        for (idx_t j = jfirst; j != jlast; j += jinc) {
            // Compute x(j) = b(j) / A(j,j), scaling x if necessary.
            real_t xj = abs1(x[j]);
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
            real_t tjj = abs1(tjjs);
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
                if constexpr (is_complex<T>) {
                    x[j] = ladiv(x[j], tjjs);
                }
                else {
                    x[j] = x[j] / tjjs;
                }
                xj = abs1(x[j]);
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
                if constexpr (is_complex<T>) {
                    x[j] = ladiv(x[j], tjjs);
                }
                else {
                    x[j] = x[j] / tjjs;
                }
                xj = abs1(x[j]);
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
                    xmax = abs1(x[i]);
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
                    xmax = max(xmax, abs1(x[i]));
                }
            }
        }
    }
    else if (trans == Op::Trans) {
        // Solve A**T * x = b
        for (idx_t j = jfirst; j != jlast; j += jinc) {
            // Compute x(j) = b(j) - sum A(k,j)*x(k).
            //                       k<>j
            real_t xj = abs1(x[j]);
            T uscal = tscal;
            real_t rec = (real_t)1 / max(xmax, (real_t)1);
            if (cnorm[j] > (bignum - xj) * rec) {
                // If x(j) could overflow, scale x by 1/(2*XMAX).
                rec = rec * (real_t)0.5;
                T tjjs;
                if (diag == Diag::NonUnit) {
                    tjjs = A(j, j) * tscal;
                }
                else {
                    tjjs = tscal;
                }
                real_t tjj = abs1(tjjs);
                if (tjj > (real_t)1) {
                    // Divide by A(j,j) when scaling x if A(j,j) > 1.
                    rec = min((real_t)1, rec * tjj);
                    if constexpr (is_complex<T>) {
                        uscal = ladiv(uscal, tjjs);
                    }
                    else {
                        uscal = uscal / tjjs;
                    }
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

            if (uscal == (T)tscal) {
                // Compute x(j) := ( x(j) - sumj ) / A(j,j) if 1/A(j,j)
                // was not used to scale the dotproduct.
                x[j] = x[j] - sumj;
                xj = abs1(x[j]);
                T tjjs;
                if (diag == Diag::NonUnit) {
                    tjjs = A(j, j) * tscal;
                }
                else {
                    tjjs = tscal;
                    if (tscal == (real_t)1) {
                        // TODO: fix goto
                    }
                }
                // Compute x(j) = x(j) / A(j,j), scaling if necessary.
                real_t tjj = abs1(tjjs);
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
                    if constexpr (is_complex<T>) {
                        x[j] = ladiv(x[j], tjjs);
                    }
                    else {
                        x[j] = x[j] / tjjs;
                    }
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
                    if constexpr (is_complex<T>) {
                        x[j] = ladiv(x[j], tjjs);
                    }
                    else {
                        x[j] = x[j] / tjjs;
                    }
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
                if constexpr (is_complex<T>) {
                    x[j] = ladiv(x[j], tjjs) - sumj;
                }
                else {
                    x[j] = x[j] / tjjs - sumj;
                }
            }
            xmax = max(xmax, abs1(x[j]));
        }
    }
    else {
        // Solve A**H * x = b
        for (idx_t j = jfirst; j != jlast; j += jinc) {
            // Compute x(j) = b(j) - sum A(k,j)*x(k).
            //                       k<>j
            real_t xj = abs1(x[j]);
            T uscal = tscal;
            real_t rec = (real_t)1 / max(xmax, (real_t)1);
            if (cnorm[j] > (bignum - xj) * rec) {
                // If x(j) could overflow, scale x by 1/(2*XMAX).
                rec = rec * (real_t)0.5;
                T tjjs;
                if (diag == Diag::NonUnit) {
                    tjjs = A(j, j) * tscal;
                }
                else {
                    tjjs = tscal;
                }
                real_t tjj = abs1(tjjs);
                if (tjj > (real_t)1) {
                    // Divide by A(j,j) when scaling x if A(j,j) > 1.
                    rec = min((real_t)1, rec * tjj);
                    if constexpr (is_complex<T>) {
                        uscal = ladiv(uscal, tjjs);
                    }
                    else {
                        uscal = uscal / tjjs;
                    }
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
                    sumj = dot(slice(col(A, j), range(0, j)),
                               slice(x, range(0, j)));
                }
                else {
                    sumj = dot(slice(col(A, j), range(j + 1, n)),
                               slice(x, range(j + 1, n)));
                }
            }
            else {
                // Otherwise, use in-line code for the dot product.
                if (uplo == Uplo::Upper) {
                    for (idx_t i = 0; i < j - 1; ++i) {
                        sumj += conj(A(i, j)) * uscal * x[i];
                    }
                }
                else {
                    for (idx_t i = j + 1; i < n; ++i) {
                        sumj += conj(A(i, j)) * uscal * x[i];
                    }
                }
            }

            if (uscal == (T)tscal) {
                // Compute x(j) := ( x(j) - sumj ) / A(j,j) if 1/A(j,j)
                // was not used to scale the dotproduct.
                x[j] = x[j] - sumj;
                xj = abs1(x[j]);
                T tjjs;
                if (diag == Diag::NonUnit) {
                    tjjs = conj(A(j, j)) * tscal;
                }
                else {
                    tjjs = tscal;
                    if (tscal == (real_t)1) {
                        // TODO: fix goto
                    }
                }
                // Compute x(j) = x(j) / A(j,j), scaling if necessary.
                real_t tjj = abs1(tjjs);
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
                    if constexpr (is_complex<T>) {
                        x[j] = ladiv(x[j], tjjs);
                    }
                    else {
                        x[j] = x[j] / tjjs;
                    }
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
                    if constexpr (is_complex<T>) {
                        x[j] = ladiv(x[j], tjjs);
                    }
                    else {
                        x[j] = x[j] / tjjs;
                    }
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
                if constexpr (is_complex<T>) {
                    x[j] = ladiv(x[j], tjjs) - sumj;
                }
                else {
                    x[j] = x[j] / tjjs - sumj;
                }
            }
            xmax = max(xmax, abs1(x[j]));
        }
    }
    scale = scale / tscal;

    // Scale the column norms by 1/TSCAL for return.
    if (tscal != (real_t)1) {
        scal((real_t)1 / tscal, cnorm);
    }
}

}  // namespace tlapack

#endif  // TLAPACK_LATRS_HELPERS_HH
