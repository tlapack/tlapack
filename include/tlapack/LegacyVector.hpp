/// @file LegacyVector.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_VECTOR_HH
#define TLAPACK_LEGACY_VECTOR_HH

#include <cassert>

#include "tlapack/base/exceptionHandling.hpp"
#include "tlapack/base/types.hpp"

namespace tlapack {

namespace internal {
    /// Auxiliary data type to vector increments.
    struct StrongOne {
        constexpr operator int() const { return 1; }
        constexpr StrongOne(int i = 1) { assert(i == 1); }
    };
}  // namespace internal

/** Legacy vector.
 *
 * @tparam T Floating-point type
 * @tparam idx_t Index type
 * @tparam D Either Direction::Forward or Direction::Backward
 */
template <
    typename T,
    class idx_t = std::size_t,
    typename int_t = internal::StrongOne,
    Direction D = Direction::Forward,
    std::enable_if_t<(D == Direction::Forward) || (D == Direction::Backward),
                     int> = 0>
struct LegacyVector {
    idx_t n;    ///< Size
    T* ptr;     ///< Pointer to array in memory
    int_t inc;  ///< Memory increment

    static constexpr Direction direction = D;

    constexpr const T& operator[](idx_t i) const noexcept
    {
        assert(i >= 0);
        assert(i < n);
        return (direction == Direction::Forward) ? *(ptr + (i * inc))
                                                 : *(ptr + ((n - 1) - i) * inc);
    }

    constexpr T& operator[](idx_t i) noexcept
    {
        assert(i >= 0);
        assert(i < n);
        return (direction == Direction::Forward) ? *(ptr + (i * inc))
                                                 : *(ptr + ((n - 1) - i) * inc);
    }

    constexpr LegacyVector(idx_t n, T* ptr, int_t inc = 1)
        : n(n), ptr(ptr), inc(inc)
    {
        tlapack_check_false(n < 0);
    }
};

}  // namespace tlapack

#endif  // TLAPACK_LEGACY_MATRIX_HH