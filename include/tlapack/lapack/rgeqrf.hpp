/// @file rgeqrf.hpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
//
// Copyright (c) 2026, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// rgeqrf utilizes geqrt3 to complete a QR factorization with a repeatedly
// halving block size as it moves to the right
//
// rgeqrf does not compute the full T matrix

#ifndef TLAPACK_RGEQRF_HH
#define TLAPACK_RGEQRF_HH

#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/geqrt3.hpp"
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/lapack/rgeqrf.hpp"

namespace tlapack {
template <TLAPACK_MATRIX matrix_a, TLAPACK_MATRIX matrix_h>

/** Computes a QR factorization of a matrix A.
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and above the diagonal of the array
 *      contain the min(m,n)-by-n upper trapezoidal matrix R
 *      (R is upper triangular if m >= n); the elements below the diagonal,
 *      with the array tau, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 *
 * @param[out] Tmatrix n-by-n matrix.
 *      On exit, contains the triangular factor of the compact WY
 *      reprensentation with the values of tau along the diagonal.
 */
void rgeqrf(matrix_a& A, matrix_h& Tmatrix)
{
    using std::size_t;
    using idx_t = size_type<matrix_a>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_a>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    idx_t nb = n;

    // // ORIGINAL DONT CHANGE
    // for (idx_t k = 0; k < n; k += nb) {
    //     // Cut block size in half with each loop if not equal to 1
    //     if (nb != 1) {
    //         nb /= 2;
    //     }

    //     // Slice Q and T
    //     auto A1 = slice(A, range(k, m), range(k, k + nb));
    //     auto A11 = slice(A, range(k, k + nb), range(k, k + nb));
    //     auto A21 = slice(A, range(k + nb, m), range(k, k + nb));
    //     auto A12 = slice(A, range(k, k + nb), range(k + nb, n));
    //     auto A22 = slice(A, range(k + nb, m), range(k + nb, n));
    //     auto T11 = slice(Tmatrix, range(k, k + nb), range(k, k + nb));
    //     auto T12 = slice(Tmatrix, range(k, k + nb), range(k + nb, n));

    //     // QR factorization of respective first panel
    //     geqrt3(A1, T11);

    //     // Copy A12 into T12 to utilize as work
    //     lacpy(Uplo::General, A12, T12);

    //     // T12 <- A11ᴴ * A12
    //     trmm(Side::Left, Uplo::Lower, Op::ConjTrans, Diag::Unit,
    //          static_cast<T>(1.0), A11, T12);

    //     // T12 <- T12 + (A21ᴴ * A22)
    //     gemm(Op::ConjTrans, Op::NoTrans, static_cast<T>(1.0), A21, A22,
    //          static_cast<T>(1.0), T12);

    //     // T12 <- T11ᴴ * T12
    //     trmm(Side::Left, Uplo::Upper, Op::ConjTrans, Diag::NonUnit,
    //          static_cast<T>(1.0), T11, T12);

    //     // A22 <- A22 - (A21 * T12)
    //     gemm(Op::NoTrans, Op::NoTrans, static_cast<T>(-1.0), A21, T12,
    //          static_cast<T>(1.0), A22);

    //     // T12 <- (A11 * T12)
    //     trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::Unit,
    //          static_cast<T>(1.0), A11, T12);

    //     // A12 <- A12 - T12
    //     for (idx_t i = 0; i < nb; ++i) {
    //         for (idx_t j = 0; j < n - k - nb; ++j) {
    //             A12(i, j) -= T12(i, j);
    //         }
    //     }
    // }
    // RECURSIVE TEST
    idx_t k = 0;
    nb = (nb + 1) / 2;

    if (k + nb == n) {
        geqrt3(A, Tmatrix);
        return;
    }

    // Slice Q and T
    auto A1 = slice(A, range(k, m), range(k, k + nb));
    auto A11 = slice(A, range(k, k + nb), range(k, k + nb));
    auto A21 = slice(A, range(k + nb, m), range(k, k + nb));
    auto A12 = slice(A, range(k, k + nb), range(k + nb, n));
    auto A22 = slice(A, range(k + nb, m), range(k + nb, n));
    auto T11 = slice(Tmatrix, range(k, k + nb), range(k, k + nb));
    auto T12 = slice(Tmatrix, range(k, k + nb), range(k + nb, n));
    auto T22 = slice(Tmatrix, range(k + nb, n), range(k + nb, n));

    // QR factorization of respective first panel
    geqrt3(A1, T11);

    // Copy A12 into T12 to utilize as work
    lacpy(Uplo::General, A12, T12);

    // T12 <- A11ᴴ * A12
    trmm(Side::Left, Uplo::Lower, Op::ConjTrans, Diag::Unit,
         static_cast<T>(1.0), A11, T12);

    // T12 <- T12 + (A21ᴴ * A22)
    gemm(Op::ConjTrans, Op::NoTrans, static_cast<T>(1.0), A21, A22,
         static_cast<T>(1.0), T12);

    // T12 <- T11ᴴ * T12
    trmm(Side::Left, Uplo::Upper, Op::ConjTrans, Diag::NonUnit,
         static_cast<T>(1.0), T11, T12);

    // A22 <- A22 - (A21 * T12)
    gemm(Op::NoTrans, Op::NoTrans, static_cast<T>(-1.0), A21, T12,
         static_cast<T>(1.0), A22);

    // T12 <- (A11 * T12)
    trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::Unit, static_cast<T>(1.0),
         A11, T12);

    // A12 <- A12 - T12
    for (idx_t i = 0; i < nb; ++i) {
        for (idx_t j = 0; j < n - k - nb; ++j) {
            A12(i, j) -= T12(i, j);
        }
    }

    rgeqrf(A22, T22);
}
}  // namespace tlapack

#endif  // TLAPACK_RGEQRF_HH
