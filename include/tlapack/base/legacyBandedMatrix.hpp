/// @file legacyBandedMatrix.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_BANDED_HH
#define TLAPACK_LEGACY_BANDED_HH

#include <cassert>

#include "tlapack/base/exceptionHandling.hpp"
#include "tlapack/base/types.hpp"

namespace tlapack {

/** Legacy banded matrix.
 *
 * Mind that the access A(i,j) is valid if, and only if,
 *  max(0,j-ku) <= i <= min(m,j+kl)
 * This class does not perform such a check,
 * otherwise it would lack in performance.
 *
 * @tparam T Floating-point type
 * @tparam idx_t Index type
 */
template <typename T, class idx_t = std::size_t>
struct legacyBandedMatrix {
    idx_t m, n, kl, ku;  ///< Sizes
    T* ptr;              ///< Pointer to array in memory

    /** Access A(i,j) = ptr[ (ku+(i-j)) + j*(ku+kl+1) ]
     *
     * Mind that this access is valid if, and only if,
     *  max(0,j-ku) <= i <= min(m,j+kl)
     * This operator does not perform such a check,
     * otherwise it would lack in performance.
     *
     */
    inline constexpr T& operator()(idx_t i, idx_t j) const noexcept
    {
        assert(i >= 0);
        assert(i < m);
        assert(j >= 0);
        assert(j < n);
        assert(j <= i + ku);
        assert(i <= j + kl);
        return ptr[(ku + i) + j * (ku + kl)];
    }

    inline constexpr legacyBandedMatrix(
        idx_t m, idx_t n, idx_t kl, idx_t ku, T* ptr)
        : m(m), n(n), kl(kl), ku(ku), ptr(ptr)
    {
        tlapack_check(m >= 0);
        tlapack_check(n >= 0);
        tlapack_check((kl + 1 <= m || m == 0) && (ku + 1 <= n || n == 0));
    }
};

}  // namespace tlapack

#endif  // TLAPACK_LEGACY_ARRAY_HH
