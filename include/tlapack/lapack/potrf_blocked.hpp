/// @file potrf_blocked.hpp Computes the Cholesky factorization of a Hermitian
/// positive definite matrix A using a blocked algorithm.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_POTRF_BLOCKED_HH
#define TLAPACK_POTRF_BLOCKED_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/potf2.hpp"

namespace tlapack {

template <TLAPACK_INDEX idx_t>
struct BlockedCholeskyOpts : public EcOpts {
    inline constexpr BlockedCholeskyOpts(const EcOpts& opts = {})
        : EcOpts(opts){};

    idx_t nb = 32;  ///< Block size
};

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A using a blocked algorithm.
 *
 * The factorization has the form
 *      $A = U^H U,$ if uplo = Upper, or
 *      $A = L L^H,$ if uplo = Lower,
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
 *      On entry, the Hermitian matrix A of size n-by-n.
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
 * @param[in] opts Options.
 *      Define the behavior of checks for NaNs.
 *
 * @return 0: successful exit.
 * @return i, 0 < i <= n, if the leading minor of order i is not
 *      positive definite, and the factorization could not be completed.
 *
 * @ingroup computational
 */
template <TLAPACK_UPLO uplo_t, TLAPACK_SMATRIX matrix_t>
int potrf_blocked(uplo_t uplo,
                  matrix_t& A,
                  const BlockedCholeskyOpts<size_type<matrix_t> >& opts)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Constants
    const real_t one(1);
    const idx_t n = nrows(A);
    const idx_t nb = opts.nb;

    // check arguments
    tlapack_check(uplo == Uplo::Lower || uplo == Uplo::Upper);
    tlapack_check(nrows(A) == ncols(A));

    // Quick return
    if (n <= 0) return 0;

    // Unblocked code
    else if (nb >= n)
        return potf2(uplo, A);

    // Blocked code
    else {
        if (uplo == Uplo::Upper) {
            for (idx_t j = 0; j < n; j += nb) {
                idx_t jb = min(nb, n - j);

                // Define AJJ and A1J
                auto AJJ = slice(A, range{j, j + jb}, range{j, j + jb});
                auto A1J = slice(A, range{0, j}, range{j, j + jb});

                herk(UPPER_TRIANGLE, CONJ_TRANS, -one, A1J, one, AJJ);

                int info = potf2(UPPER_TRIANGLE, AJJ);
                if (info != 0) {
                    tlapack_error(
                        info + j,
                        "The leading minor of the reported order is not "
                        "positive definite,"
                        " and the factorization could not be completed.");
                    return info + j;
                }

                if (j + jb < n) {
                    // Define B and C
                    auto B = slice(A, range{0, j}, range{j + jb, n});
                    auto C = slice(A, range{j, j + jb}, range{j + jb, n});

                    // Compute the current block row
                    gemm(CONJ_TRANS, NO_TRANS, -one, A1J, B, one, C);
                    trsm(LEFT_SIDE, UPPER_TRIANGLE, CONJ_TRANS, NON_UNIT_DIAG,
                         one, AJJ, C);
                }
            }
        }
        else {
            for (idx_t j = 0; j < n; j += nb) {
                idx_t jb = min(nb, n - j);

                // Define AJJ and AJ1
                auto AJJ = slice(A, range{j, j + jb}, range{j, j + jb});
                auto AJ1 = slice(A, range{j, j + jb}, range{0, j});

                herk(LOWER_TRIANGLE, NO_TRANS, -one, AJ1, one, AJJ);

                int info = potf2(LOWER_TRIANGLE, AJJ);
                if (info != 0) {
                    tlapack_error(
                        info + j,
                        "The leading minor of the reported order is not "
                        "positive definite,"
                        " and the factorization could not be completed.");
                    return info + j;
                }

                if (j + jb < n) {
                    // Define B and C
                    auto B = slice(A, range{j + jb, n}, range{0, j});
                    auto C = slice(A, range{j + jb, n}, range{j, j + jb});

                    // Compute the current block row
                    gemm(NO_TRANS, CONJ_TRANS, -one, B, AJ1, one, C);
                    trsm(RIGHT_SIDE, LOWER_TRIANGLE, CONJ_TRANS, NON_UNIT_DIAG,
                         one, AJJ, C);
                }
            }
        }

        // Report infs and nans on the output
        tlapack_warn_nans_in_matrix(opts.ec, uplo, A, n + 1,
                                    "The factorization has some nans.");
        tlapack_warn_infs_in_matrix(opts.ec, uplo, A, n + 1,
                                    "The factorization has some infs.");

        return 0;
    }
}

template <TLAPACK_UPLO uplo_t,
          TLAPACK_SMATRIX matrix_t,
          disable_if_allow_optblas_t<matrix_t> = 0>
inline int potrf_blocked(uplo_t uplo, matrix_t& A)
{
    return potrf_blocked(uplo, A, {});
}

#ifdef TLAPACK_USE_LAPACKPP

template <TLAPACK_UPLO uplo_t,
          TLAPACK_LEGACY_MATRIX matrix_t,
          enable_if_allow_optblas_t<matrix_t> = 0>
inline int potrf_blocked(uplo_t uplo, matrix_t& A)
{
    // Legacy objects
    auto A_ = legacy_matrix(A);

    // Constants to forward
    const auto& n = A_.n;

    if constexpr (layout<matrix_t> == Layout::ColMajor) {
        return ::lapack::potrf((::blas::Uplo)(Uplo)uplo, n, A_.ptr, A_.ldim);
    }
    else {
        return ::lapack::potrf(
            ((uplo == Uplo::Lower) ? ::blas::Uplo::Upper : ::blas::Uplo::Lower),
            n, A_.ptr, A_.ldim);
    }
}

#endif

}  // namespace tlapack

#endif  // TLAPACK_POTRF_BLOCKED_HH
