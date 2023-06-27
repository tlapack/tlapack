/// @file potf2.hpp Computes the Cholesky factorization of a Hermitian positive
/// definite matrix A using a level-2 algorithm.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_POTF2_HH
#define TLAPACK_POTF2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/dot.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/scal.hpp"

namespace tlapack {

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A using a level-2 algorithm.
 *
 * The factorization has the form
 *     $A = U^H U,$ if uplo = Upper, or
 *     $A = L L^H,$ if uplo = Lower,
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * @tparam uplo_t
 *      Access type: Upper or Lower.
 *      Either Uplo or any class that implements `operator Uplo()`.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *      On entry, the Hermitian matrix A.
 *
 *      - If uplo = Uplo::Upper, the strictly lower
 *      triangular part of A is not referenced.
 *
 *      - If uplo = Uplo::Lower, the strictly upper
 *      triangular part of A is not referenced.
 *
 *      - On successful exit, the factor U or L from the Cholesky
 *      factorization $A = U^H U$ or $A = L L^H.$
 *
 * @return = 0: successful exit
 * @return i, 0 < i <= n, if the leading minor of order i is not
 *     positive definite, and the factorization could not be completed.
 *
 * @ingroup computational
 */
template <TLAPACK_UPLO uplo_t,
          TLAPACK_SMATRIX matrix_t,
          disable_if_allow_optblas_t<matrix_t> = 0>
int potf2(uplo_t uplo, matrix_t& A)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // Constants
    const real_t one(1);
    const real_t zero(0);
    const idx_t n = nrows(A);

    // check arguments
    tlapack_check(uplo == Uplo::Lower || uplo == Uplo::Upper);
    tlapack_check(nrows(A) == ncols(A));

    // Quick return
    if (n <= 0) return 0;

    if (uplo == Uplo::Upper) {
        // Compute the Cholesky factorization A = U^H * U
        for (idx_t j = 0; j < n; ++j) {
            auto colj = slice(A, pair{0, j}, j);

            // Compute U(j,j) and test for non-positive-definiteness
            real_t ajj = real(A(j, j)) - real(dot(colj, colj));
            if (ajj > zero) {
                ajj = sqrt(ajj);
                A(j, j) = T(ajj);
            }
            else {
                tlapack_error(
                    j + 1,
                    "The leading minor of order j+1 is not positive definite,"
                    " and the factorization could not be completed.");
                return j + 1;
            }

            // Compute elements j+1:n of row j
            if (j + 1 < n) {
                auto Ajj = slice(A, pair{0, j}, pair{j + 1, n});
                auto rowj = slice(A, j, pair{j + 1, n});

                // rowj := rowj - conj(colj) * Ajj
                for (idx_t i = 0; i < j; ++i)
                    colj[i] = conj(colj[i]);
                gemv(Transpose, -one, Ajj, colj, one, rowj);
                for (idx_t i = 0; i < j; ++i)
                    colj[i] = conj(colj[i]);

                /// TODO: replace by rscl when available
                scal(one / ajj, rowj);
            }
        }
    }
    else {
        // Compute the Cholesky factorization A = L * L^H
        for (idx_t j = 0; j < n; ++j) {
            auto rowj = slice(A, j, pair{0, j});

            // Compute L(j,j) and test for non-positive-definiteness
            real_t ajj = real(A(j, j)) - real(dot(rowj, rowj));
            if (ajj > zero) {
                ajj = sqrt(ajj);
                A(j, j) = T(ajj);
            }
            else {
                tlapack_error(
                    j + 1,
                    "The leading minor of order j+1 is not positive definite,"
                    " and the factorization could not be completed.");
                return j + 1;
            }

            // Compute elements j+1:n of column j
            if (j + 1 < n) {
                auto Ajj = slice(A, pair{j + 1, n}, pair{0, j});
                auto colj = slice(A, pair{j + 1, n}, j);

                // colj := colj - Ajj * conj(rowj)
                for (idx_t i = 0; i < j; ++i)
                    rowj[i] = conj(rowj[i]);
                gemv(noTranspose, -one, Ajj, rowj, one, colj);
                for (idx_t i = 0; i < j; ++i)
                    rowj[i] = conj(rowj[i]);

                /// TODO: replace by rscl when available
                scal(one / ajj, colj);
            }
        }
    }

    return 0;
}

#ifdef USE_LAPACKPP_WRAPPERS

template <TLAPACK_UPLO uplo_t,
          TLAPACK_SMATRIX matrix_t,
          enable_if_allow_optblas_t<matrix_t> = 0>
int potf2(uplo_t uplo, matrix_t& A)
{
    // Legacy objects
    auto A_ = legacy_matrix(A);

    // Constants to forward
    const auto& n = A_.n;

    if constexpr (layout<matrix_t> == Layout::ColMajor) {
        return ::lapack::potf2((::blas::Uplo)(Uplo)uplo, n, A_.ptr, A_.ldim);
    }
    else {
        return ::lapack::potf2(
            ((uplo == Uplo::Lower) ? ::blas::Uplo::Upper : ::blas::Uplo::Lower),
            n, A_.ptr, A_.ldim);
    }
}

#endif

}  // namespace tlapack

#endif  // TLAPACK_POTF2_HH
