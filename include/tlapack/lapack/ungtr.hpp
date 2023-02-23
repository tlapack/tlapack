/// @file ungtr.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGTR_HH
#define TLAPACK_UNGTR_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/ung2l.hpp"
#include "tlapack/lapack/ung2r.hpp"

namespace tlapack {

/**
 * @brief Workspace query for ungtr().
 *
 * @tparam uplo_t Either Uplo or any class that implements `operator Uplo()`.
 *
 * @param[in] uplo
 *      - Uplo::Upper:   Upper triangle of Q contains elementary reflectors;
 *      - Uplo::Lower:   Lower triangle of Q contains elementary reflectors;
 *
 * @param[in] Q n-by-n matrix.
 *
 * @param[in] tau Vector of length k.
 *     The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template <class matrix_t, class vector_t, class uplo_t>
inline constexpr void ungtr_worksize(uplo_t uplo,
                                     const matrix_t& Q,
                                     const vector_t& tau,
                                     workinfo_t& workinfo,
                                     const workspace_opts_t<>& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(Q);

    if (uplo == Uplo::Lower) {
        auto Qrefl = slice(Q, pair(1, n), pair(1, n));
        ung2r_worksize(size(tau), Qrefl, tau, opts);
    }
    else {
        auto Qrefl = slice(Q, pair(0, n - 1), pair(0, n - 1));
        ung2l_worksize(Qrefl, tau, opts);
    }
}

/**
 * @brief Generates a real orthogonal matrix Q which is defined as the product
 * of k elementary reflectors of order n, as returned by hetrd.
 *
 * @tparam uplo_t Either Uplo or any class that implements `operator Uplo()`.
 *
 * @param[in] uplo
 *      - Uplo::Upper:   Upper triangle of Q contains elementary reflectors;
 *      - Uplo::Lower:   Lower triangle of Q contains elementary reflectors;
 *
 * @param[in,out] Q n-by-n matrix.
 *      On entry, the vectors which define the elementary reflectors, as
 *      returned by hetrd.
 *      On exit, the n-by-n orthogonal matrix Q.
 *
 * @param[in] tau Vector of length k.
 *     The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      @c opts.work is used if whenever it has sufficient size.
 *      The sufficient size can be obtained through a workspace query.
 *
 * @return 0 if success
 *
 * @ingroup computational
 */
template <class matrix_t, class vector_t, class uplo_t>
int ungtr(uplo_t uplo,
          matrix_t& Q,
          const vector_t& tau,
          const workspace_opts_t<>& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(Q);
    const real_t one(1);
    const real_t zero(0);

    if (uplo == Uplo::Lower) {
        // Move the reflectors in Q
        for (idx_t j = n - 2; j != idx_t(0); --j)
            for (idx_t i = j + 1; i < n; ++i)
                Q(i, j) = Q(i, j - 1);

        // Complete Q with the identity
        for (idx_t i = 1; i < n; ++i) {
            Q(i, 0) = zero;
            Q(0, i) = zero;
        }
        Q(0, 0) = one;

        // Compute the Q part that use the reflectors
        auto Qrefl = slice(Q, pair(1, n), pair(1, n));
        return ung2r(size(tau), Qrefl, tau, opts);
    }
    else {
        // Move the reflectors in Q
        for (idx_t j = 1; j < n - 1; ++j)
            for (idx_t i = 0; i < j; ++i)
                Q(i, j) = Q(i, j + 1);

        // Complete Q with the identity
        for (idx_t i = 0; i < n - 1; ++i) {
            Q(i, n - 1) = zero;
            Q(n - 1, i) = zero;
        }
        Q(n - 1, n - 1) = one;

        // Compute the Q part that use the reflectors
        auto Qrefl = slice(Q, pair(0, n - 1), pair(0, n - 1));
        return ung2l(Qrefl, tau, opts);
    }
}

}  // namespace tlapack

#endif  // TLAPACK_UNGTR_HH