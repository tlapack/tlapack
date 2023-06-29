/// @file legacyArray.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_ARRAY_HH
#define TLAPACK_LEGACY_ARRAY_HH

#include <cassert>

#include "tlapack/base/exceptionHandling.hpp"
#include "tlapack/base/legacyBandedMatrix.hpp"
#include "tlapack/base/types.hpp"

namespace tlapack {

/** Legacy matrix.
 *
 * legacyMatrix::ldim is assumed to be positive.
 *
 * @tparam T Floating-point type
 * @tparam idx_t Index type
 * @tparam L Either Layout::ColMajor or Layout::RowMajor
 */
template <class T,
          class idx_t = std::size_t,
          Layout L = Layout::ColMajor,
          std::enable_if_t<(L == Layout::RowMajor) || (L == Layout::ColMajor),
                           int> = 0>
struct legacyMatrix {
    idx_t m, n;  ///< Sizes
    T* ptr;      ///< Pointer to array in memory
    idx_t ldim;  ///< Leading dimension

    static constexpr Layout layout = L;

    inline constexpr const T& operator()(idx_t i, idx_t j) const noexcept
    {
        assert(i >= 0);
        assert(i < m);
        assert(j >= 0);
        assert(j < n);
        return (layout == Layout::ColMajor) ? ptr[i + j * ldim]
                                            : ptr[i * ldim + j];
    }

    inline constexpr T& operator()(idx_t i, idx_t j) noexcept
    {
        assert(i >= 0);
        assert(i < m);
        assert(j >= 0);
        assert(j < n);
        return (layout == Layout::ColMajor) ? ptr[i + j * ldim]
                                            : ptr[i * ldim + j];
    }

    inline constexpr legacyMatrix(idx_t m, idx_t n, T* ptr, idx_t ldim)
        : m(m), n(n), ptr(ptr), ldim(ldim)
    {
        tlapack_check(m >= 0);
        tlapack_check(n >= 0);
        tlapack_check(ldim >= ((layout == Layout::ColMajor) ? m : n));
    }

    inline constexpr legacyMatrix(idx_t m, idx_t n, T* ptr)
        : m(m), n(n), ptr(ptr), ldim((layout == Layout::ColMajor) ? m : n)
    {
        tlapack_check(m >= 0);
        tlapack_check(n >= 0);
    }
};

/** Legacy vector.
 *
 * @tparam T Floating-point type
 * @tparam idx_t Index type
 * @tparam D Either Direction::Forward or Direction::Backward
 */
template <typename T,
          class idx_t = std::size_t,
          typename int_t = internal::StrongOne,
          Direction D = Direction::Forward>
struct legacyVector {
    idx_t n;    ///< Size
    T* ptr;     ///< Pointer to array in memory
    int_t inc;  ///< Memory increment

    static constexpr Direction direction = D;

    inline constexpr const T& operator[](idx_t i) const noexcept
    {
        assert(i >= 0);
        assert(i < n);
        return (direction == Direction::Forward) ? *(ptr + (i * inc))
                                                 : *(ptr + ((n - 1) - i) * inc);
    }

    inline constexpr T& operator[](idx_t i) noexcept
    {
        assert(i >= 0);
        assert(i < n);
        return (direction == Direction::Forward) ? *(ptr + (i * inc))
                                                 : *(ptr + ((n - 1) - i) * inc);
    }

    inline constexpr legacyVector(idx_t n, T* ptr, int_t inc = 1)
        : n(n), ptr(ptr), inc(inc)
    {
        tlapack_check_false(n < 0);
    }
};

}  // namespace tlapack

#endif  // TLAPACK_LEGACY_ARRAY_HH
