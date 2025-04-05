/// @file ungtr.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGTR_HH
#define TLAPACK_UNGTR_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/ungql.hpp"
#include "tlapack/lapack/ungqr.hpp"

namespace tlapack {

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
 * @return 0 if success
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX Q_t, TLAPACK_SVECTOR tau_t, class uplo_t>
int ungtr(uplo_t uplo, Q_t& Q, const tau_t& tau)
{
    using T = type_t<Q_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<Q_t>;
    using pair = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(Q);
    const real_t one(1);
    const real_t zero(0);

    // Quick return if possible
    if (n == 0) return 0;

    if (uplo == Uplo::Lower) {
        // Move the reflectors in Q
        for (idx_t j2 = n - 1; j2 > 1; --j2) {
            idx_t j = j2 - 1;
            for (idx_t i = j + 1; i < n; ++i)
                Q(i, j) = Q(i, j - 1);
        }

        // Complete Q with the identity
        for (idx_t i = 1; i < n; ++i) {
            Q(i, 0) = zero;
            Q(0, i) = zero;
        }
        Q(0, 0) = one;

        // Compute the Q part that use the reflectors
        auto Qrefl = slice(Q, pair(1, n), pair(1, n));
        // Todo: use workspace
        return ungqr(Qrefl, tau);
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
        // Todo: use workspace
        return ungql(Qrefl, tau);
    }
}

}  // namespace tlapack

#endif  // TLAPACK_UNGTR_HH