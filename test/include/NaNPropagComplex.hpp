/// @file test/include/NaNPropagComplex.hpp
/// @brief std::complex specialization that propagates NaNs.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TEST_COMPLEX_HH
#define TLAPACK_TEST_COMPLEX_HH

#include <tlapack/base/types.hpp>

namespace tlapack {

template <class T>
struct NaNPropagComplex : public std::complex<T> {
    constexpr NaNPropagComplex(const T& r = T(), const T& i = T())
        : std::complex<T>(r, i)
    {}

    // operators:

    template <typename U>
    constexpr NaNPropagComplex& operator=(const std::complex<U>& x)
    {
        (std::complex<T>&)(*this) = x;
        return *this;
    }

    template <typename U>
    constexpr NaNPropagComplex& operator+=(const std::complex<U>& x)
    {
        if (isnan(x))
            *this = std::numeric_limits<T>::quiet_NaN();
        else if (isnan(*this))
            *this = std::numeric_limits<T>::signaling_NaN();
        else
            (std::complex<T>&)(*this) += x;

        return *this;
    }

    template <typename U>
    constexpr NaNPropagComplex& operator-=(const std::complex<U>& x)
    {
        if (isnan(x))
            *this = std::numeric_limits<T>::quiet_NaN();
        else if (isnan(*this))
            *this = std::numeric_limits<T>::signaling_NaN();
        else
            (std::complex<T>&)(*this) -= x;

        return *this;
    }

    template <typename U>
    constexpr NaNPropagComplex& operator*=(const std::complex<U>& x)
    {
        if (isnan(x))
            *this = std::numeric_limits<T>::quiet_NaN();
        else if (isnan(*this))
            *this = std::numeric_limits<T>::signaling_NaN();
        else
            (std::complex<T>&)(*this) *= x;

        return *this;
    }

    constexpr NaNPropagComplex& operator*=(const T& x)
    {
        (std::complex<T>&)(*this) *= x;
        return *this;
    }

    template <typename U>
    constexpr NaNPropagComplex& operator/=(const std::complex<U>& x)
    {
        if (isnan(x))
            *this = std::numeric_limits<T>::quiet_NaN();
        else if (isnan(*this))
            *this = std::numeric_limits<T>::signaling_NaN();
        else
            (std::complex<T>&)(*this) /= x;

        return *this;
    }

    constexpr NaNPropagComplex& operator/=(const T& x)
    {
        (std::complex<T>&)(*this) /= x;
        return *this;
    }

    // wrappers:

    friend constexpr NaNPropagComplex operator+(const NaNPropagComplex& x,
                                                const NaNPropagComplex& y)
    {
        NaNPropagComplex r = x;
        r += y;
        return r;
    }

    friend constexpr NaNPropagComplex operator+(const T& x,
                                                const NaNPropagComplex& y)
    {
        NaNPropagComplex r = y;
        r += x;
        return r;
    }

    friend constexpr NaNPropagComplex operator+(const NaNPropagComplex& x,
                                                const T& y)
    {
        NaNPropagComplex r = x;
        r += y;
        return r;
    }

    friend constexpr NaNPropagComplex operator-(const NaNPropagComplex& x,
                                                const NaNPropagComplex& y)
    {
        NaNPropagComplex r = x;
        r -= y;
        return r;
    }

    friend constexpr NaNPropagComplex operator-(const T& x,
                                                const NaNPropagComplex& y)
    {
        NaNPropagComplex r = y;
        r -= x;
        return r;
    }

    friend constexpr NaNPropagComplex operator-(const NaNPropagComplex& x,
                                                const T& y)
    {
        NaNPropagComplex r = x;
        r -= y;
        return r;
    }

    friend constexpr NaNPropagComplex operator*(const NaNPropagComplex& x,
                                                const NaNPropagComplex& y)
    {
        NaNPropagComplex r = x;
        r *= y;
        return r;
    }

    friend constexpr NaNPropagComplex operator*(const T& x,
                                                const NaNPropagComplex& y)
    {
        NaNPropagComplex r = y;
        r *= x;
        return r;
    }

    friend constexpr NaNPropagComplex operator*(const NaNPropagComplex& x,
                                                const T& y)
    {
        NaNPropagComplex r = x;
        r *= y;
        return r;
    }

    friend constexpr NaNPropagComplex operator/(const NaNPropagComplex& x,
                                                const NaNPropagComplex& y)
    {
        NaNPropagComplex r = x;
        r /= y;
        return r;
    }

    friend constexpr NaNPropagComplex operator/(const T& x,
                                                const NaNPropagComplex& y)
    {
        NaNPropagComplex r = x;
        r /= y;
        return r;
    }

    friend constexpr NaNPropagComplex operator/(const NaNPropagComplex& x,
                                                const T& y)
    {
        NaNPropagComplex r = x;
        r /= y;
        return r;
    }

    // Other math operations:

    /** 2-norm absolute value, sqrt( |Re(x)|^2 + |Im(x)|^2 )
     *
     * Note that std::abs< std::complex > does not overflow or underflow at
     * intermediate stages of the computation.
     * @see https://en.cppreference.com/w/cpp/numeric/complex/abs
     *
     * However, std::abs(std::complex<T>) may not propagate NaNs. See
     * https://github.com/tlapack/tlapack/issues/134#issue-1364091844.
     * Operations with `std::complex<T>`, for `T=float,double,long double` are
     * wrappers to operations in C. Other types have their implementation in
     * C++. Because of that, the logic of complex multiplication, division and
     * other operations may change from type to type. See
     * https://github.com/advanpix/mpreal/issues/11.
     *
     * Also, std::abs< mpfr::mpreal > may not propagate Infs.
     */
    friend inline T abs(const NaNPropagComplex& x)
    {
        if (isnan(real(x)) || isnan(imag(x)))
            return std::numeric_limits<T>::quiet_NaN();
        else if (isinf(real(x)) || isinf(imag(x)))
            return std::numeric_limits<T>::infinity();
        else
            return abs((const std::complex<T>&)x);
    }

    friend inline NaNPropagComplex conj(const NaNPropagComplex& x)
    {
        return conj((const std::complex<T>&)x);
    }
};

namespace traits {
    template <typename T>
    struct real_type_traits<NaNPropagComplex<T>, int>
        : public real_type_traits<std::complex<T>, int> {};

    template <typename T>
    struct complex_type_traits<NaNPropagComplex<T>, int>
        : public complex_type_traits<std::complex<T>, int> {};
}  // namespace traits

}  // namespace tlapack

#endif  // TLAPACK_TEST_COMPLEX_HH