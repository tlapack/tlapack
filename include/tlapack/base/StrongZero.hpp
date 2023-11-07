/// @file base/StrongZero.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BASE_STRONGZERO_HH
#define TLAPACK_BASE_STRONGZERO_HH

#include <cassert>
#include <cstdint>
#include <limits>

#include "tlapack/base/scalar_type_traits.hpp"

namespace tlapack {

/**
 * @brief Strong zero type. Used to enforce the BLAS behavior of ignoring part
 * of the input when a coefficient is zero.
 *
 * @note this type satisfies the concepts: tlapack::concepts::Scalar,
 * tlapack::concepts::Real, and tlapack::concepts::Complex.
 *
 * Suppose x is of type T, and z is of type StrongZero. Then:
 *
 * 1. z is equivalent to T(0).
 * 2. x += z and x -= z do not modify x.
 * 3. x *= z is equivalent to x = T(0).
 * 4. x /= z is equivalent to setting x to infinity.
 * 5. x + z, x - z and z + x return x.
 * 6. x - z returns -x.
 * 7. x * z and z * x return z.
 * 8. x / z returns infinity.
 * 9. z / x returns z.
 * 10. z * z returns a quiet NaN.
 *
 */
struct StrongZero {
    // Constructors

    constexpr StrongZero() {}

    template <typename T>
    constexpr StrongZero(const T& x) noexcept
    {
        assert(x == T(0));
    }

    template <typename T, typename U>
    constexpr StrongZero(const T& x, const U& y) noexcept
    {
        assert(x == T(0));
        assert(y == U(0));
    }

    // Conversion operators

    template <typename T>
    explicit constexpr operator T() const noexcept
    {
        return T(0);
    }

    // Assignment operators

    template <typename T>
    friend constexpr T& operator+=(T& lhs, StrongZero) noexcept
    {
        return lhs;
    }

    template <typename T>
    friend constexpr T& operator-=(T& lhs, StrongZero) noexcept
    {
        return lhs;
    }

    template <typename T>
    friend constexpr T& operator*=(T& lhs, StrongZero) noexcept
    {
        return (lhs = T(0));
    }

    template <typename T>
    friend constexpr T& operator/=(T& lhs, StrongZero) noexcept
    {
        return (lhs = std::numeric_limits<T>::infinity());
    }

    // Arithmetic operators

    friend constexpr StrongZero operator+(StrongZero, StrongZero) noexcept
    {
        return StrongZero();
    }

    template <typename T>
    friend constexpr T operator+(StrongZero, const T& rhs) noexcept
    {
        return rhs;
    }

    template <typename T>
    friend constexpr T operator+(const T& lhs, StrongZero) noexcept
    {
        return lhs;
    }

    friend constexpr StrongZero operator-(StrongZero, StrongZero) noexcept
    {
        return StrongZero();
    }

    template <typename T>
    friend constexpr T operator-(StrongZero, const T& rhs) noexcept
    {
        return -rhs;
    }

    template <typename T>
    friend constexpr T operator-(const T& lhs, StrongZero) noexcept
    {
        return lhs;
    }

    friend constexpr StrongZero operator*(StrongZero, StrongZero) noexcept
    {
        return StrongZero();
    }

    template <typename T>
    friend constexpr StrongZero operator*(StrongZero, const T&) noexcept
    {
        return StrongZero();
    }

    template <typename T>
    friend constexpr StrongZero operator*(const T&, StrongZero) noexcept
    {
        return StrongZero();
    }

    friend constexpr float operator/(StrongZero, StrongZero) noexcept
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    template <typename T>
    friend constexpr StrongZero operator/(StrongZero, const T&) noexcept
    {
        return StrongZero();
    }

    template <typename T>
    friend constexpr T operator/(const T&, StrongZero) noexcept
    {
        return std::numeric_limits<T>::infinity();
    }

    // Change of sign

    constexpr StrongZero operator-() const noexcept { return StrongZero(); }

    // Comparison operators

    constexpr bool operator==(StrongZero) const noexcept { return true; }
    constexpr bool operator!=(StrongZero) const noexcept { return false; }
    constexpr bool operator>(StrongZero) const noexcept { return false; }
    constexpr bool operator<(StrongZero) const noexcept { return false; }
    constexpr bool operator>=(StrongZero) const noexcept { return true; }
    constexpr bool operator<=(StrongZero) const noexcept { return true; }
    friend constexpr bool isinf(StrongZero) noexcept { return false; }
    friend constexpr bool isnan(StrongZero) noexcept { return false; }

    // Math functions

    friend constexpr StrongZero abs(StrongZero) noexcept
    {
        return StrongZero();
    }
    friend constexpr StrongZero sqrt(StrongZero) noexcept
    {
        return StrongZero();
    }
    friend constexpr int pow(int, StrongZero) noexcept { return 1; }
    friend constexpr float log2(StrongZero) noexcept
    {
        return -std::numeric_limits<float>::infinity();
    }
    friend constexpr StrongZero ceil(StrongZero) noexcept
    {
        return StrongZero();
    }
    friend constexpr StrongZero floor(StrongZero) noexcept
    {
        return StrongZero();
    }
};

namespace traits {

    // for either StrongZero, return the other type
    template <typename T>
    struct real_type_traits<StrongZero, T, int> : real_type_traits<T, int> {};

    // for either StrongZero, return the other type
    template <typename T>
    struct real_type_traits<T, StrongZero, int> : real_type_traits<T, int> {};

    // tlapack::real_type<StrongZero> is StrongZero
    template <>
    struct real_type_traits<StrongZero, int> {
        using type = StrongZero;
        constexpr static bool is_real = true;
    };

    // for either StrongZero, return the other type
    template <typename T>
    struct complex_type_traits<StrongZero, T, int>
        : complex_type_traits<T, int> {};

    // for either StrongZero, return the other type
    template <typename T>
    struct complex_type_traits<T, StrongZero, int>
        : complex_type_traits<T, int> {};

    // tlapack::complex_type<StrongZero> is StrongZero
    template <>
    struct complex_type_traits<StrongZero, int> {
        using type = StrongZero;
        constexpr static bool is_complex = false;
    };
}  // namespace traits

}  // namespace tlapack

template <>
struct std::numeric_limits<tlapack::StrongZero> {
    static constexpr bool is_specialized = true;

    static constexpr tlapack::StrongZero min() noexcept
    {
        return tlapack::StrongZero();
    }
    static constexpr tlapack::StrongZero max() noexcept
    {
        return tlapack::StrongZero();
    }
    static constexpr tlapack::StrongZero lowest() noexcept
    {
        return tlapack::StrongZero();
    }
    static constexpr tlapack::StrongZero epsilon() noexcept
    {
        return tlapack::StrongZero();
    }
    static constexpr tlapack::StrongZero round_error() noexcept
    {
        return tlapack::StrongZero();
    }
    static constexpr tlapack::StrongZero infinity() noexcept
    {
        return tlapack::StrongZero();
    }
    static constexpr tlapack::StrongZero quiet_NaN() noexcept
    {
        return tlapack::StrongZero();
    }
    static constexpr tlapack::StrongZero signaling_NaN() noexcept
    {
        return tlapack::StrongZero();
    }
    static constexpr tlapack::StrongZero denorm_min() noexcept
    {
        return tlapack::StrongZero();
    }
};

#endif  // TLAPACK_BASE_STRONGZERO_HH
