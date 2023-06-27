/// @file potrf_blocked_right_looking.hpp Computes the Cholesky factorization of
/// a Hermitian positive definite matrix A using the right-looking blocked
/// algorithm.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_POTRF_BLOCKED_RL_HH
#define TLAPACK_POTRF_BLOCKED_RL_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/potf2.hpp"
#include "tlapack/lapack/potrf_blocked.hpp"

namespace tlapack {

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
int potrf_rl(uplo_t uplo,
             matrix_t& A,
             const potrf_blocked_opts_t<size_type<matrix_t> >& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    using std::min;

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

                // Define AJJ
                auto AJJ = slice(A, pair{j, j + jb}, pair{j, j + jb});

                int info = potf2(uplo, AJJ);
                if (info != 0) {
                    tlapack_error(
                        info + j,
                        "The leading minor of the reported order is not "
                        "positive definite,"
                        " and the factorization could not be completed.");
                    return info + j;
                }

                if (j + jb < n) {
                    auto B = slice(A, pair{j, j + jb}, pair{j + jb, n});
                    auto C = slice(A, pair{j + jb, n}, pair{j + jb, n});

                    trsm(left_side, uplo, conjTranspose, nonUnit_diagonal, one,
                         AJJ, B);
                    herk(uplo, conjTranspose, -one, B, one, C);
                }
            }
        }
        else {
            for (idx_t j = 0; j < n; j += nb) {
                idx_t jb = min(nb, n - j);

                // Define AJJ
                auto AJJ = slice(A, pair{j, j + jb}, pair{j, j + jb});

                int info = potf2(uplo, AJJ);
                if (info != 0) {
                    tlapack_error(
                        info + j,
                        "The leading minor of the reported order is not "
                        "positive definite,"
                        " and the factorization could not be completed.");
                    return info + j;
                }

                if (j + jb < n) {
                    auto B = slice(A, pair{j + jb, n}, pair{j, j + jb});
                    auto C = slice(A, pair{j + jb, n}, pair{j + jb, n});

                    trsm(right_side, uplo, conjTranspose, nonUnit_diagonal, one,
                         AJJ, B);
                    herk(uplo, noTranspose, -one, B, one, C);
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

}  // namespace tlapack

#endif  // TLAPACK_POTRF_BLOCKED_RL_HH
