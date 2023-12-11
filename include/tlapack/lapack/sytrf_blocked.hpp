/// @file sytrf_blocked.hpp Computes the Bunch-Kaufman factorization of a
/// symmetric matrix A using a blocked algorithm.
/// @author Hugh M Kadhem, University of California Berkeley, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_SYTRF_BLOCKED_HH
#define TLAPACK_SYTRF_BLOCKED_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/sytf3.hpp"

namespace tlapack {

/** Worspace query of sytrf_blocked()
 *
 * @param[in] A n-by-n matrix.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t>
constexpr WorkInfo sytrf_blocked_worksize(const matrix_t& A,
                                          const BlockedLDLOpts& opts)
{
    if constexpr (is_same_v<T, type_t<matrix_t>>)
        return WorkInfo(nrows(A) * opts.nb);
    else
        return WorkInfo(0);
}

/** @copybrief sytrf_work()
 * @copydetails sytrf()
 * @ingroup computational
 */
template <TLAPACK_UPLO uplo_t,
          TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR ipiv_t,
          TLAPACK_WORKSPACE work_t>
int sytrf_blocked_work(uplo_t uplo,
                       matrix_t& A,
                       ipiv_t& ipiv,
                       work_t& work,
                       const BlockedLDLOpts& opts)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Constants
    const idx_t n = nrows(A);
    const idx_t nb = opts.nb;

    // check arguments
    tlapack_check(uplo == Uplo::Lower || uplo == Uplo::Upper);
    tlapack_check(nrows(A) == ncols(A));

    // Quick return
    if (n <= 0) return 0;
    int info = 0;
    if (uplo == Uplo::Upper) {
        int j = (int)(n)-1;
        while (j > 0) {
            int jn = min((int)(nb), j + 1);
            int step = jn;
            auto Aj = slice(A, range{0, j + 1}, range{0, j + 1});
            auto ipivj = slice(ipiv, range{0, j + 1});
            int infoj = tlapack::sytf3(uplo, Aj, ipivj, work, opts);
            info = info == 0 ? infoj : info;
            if (ipiv[j + 1 - jn] >= j + 1) {
                --step;
            }
            j -= step;
        }
        if (ipiv[0] >= 0) ipiv[0] = 0;
    }
    else if (uplo == Uplo::Lower) {
        int j = 0;
        while (j < n - 1) {
            int jn = min((int)(nb), (int)(n)-j);
            int step = jn;
            auto Aj = slice(A, range{j, n}, range{j, n});
            auto ipivj = slice(ipiv, range{j, n});
            int infoj = tlapack::sytf3(uplo, Aj, ipivj, work, opts);
            if (infoj > 0) infoj += j;
            info = info == 0 ? infoj : info;
            if (ipivj[jn - 1] >= ((int)(n)-j)) {
                --step;
            }
            for (int k = 0; k < step; ++k) {
                if (ipivj[k] >= 0) {
                    ipivj[k] += j;
                }
                else {
                    ipivj[k] -= j;
                    ipivj[++k] -= j;
                }
            }
            j += step;
        }
        if (ipiv[n - 1] >= 0) ipiv[n - 1] = n - 1;
    }
    return info;
}

/** @copybrief sytrf()
 * @copydetails sytrf()
 * @ingroup alloc_workspace
 */
template <TLAPACK_UPLO uplo_t, TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR ipiv_t>
int sytrf_blocked(uplo_t uplo,
                  matrix_t& A,
                  ipiv_t& ipiv,
                  const BlockedLDLOpts& opts)
{
    using T = type_t<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // Allocates workspace
    WorkInfo workinfo = sytrf_blocked_worksize<T>(A, opts);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return sytrf_blocked_work(uplo, A, ipiv, work, opts);
}

template <TLAPACK_UPLO uplo_t, TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR ipiv_t>
int sytrf_blocked(uplo_t uplo, matrix_t& A, ipiv_t& ipiv)
{
    return sytrf_blocked(uplo, A, ipiv, {});
}

}  // namespace tlapack

#endif  // TLAPACK_POTRF_BLOCKED_HH
