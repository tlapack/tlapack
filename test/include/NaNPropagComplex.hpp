/// @file test/include/NaNPropagComplex.hpp
/// @brief std::complex specialization that propagates NaNs.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TEST_COMPLEX_HPP
#define TLAPACK_TEST_COMPLEX_HPP

#include <tlapack/base/types.hpp>

namespace tlapack {

template <class T>
struct NaNPropagComplex : public std::complex<T> {
    constexpr NaNPropagComplex(const T& r = T(), const T& i = T())
        : std::complex<T>(r, i)
    {}

    // operators:

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

    // wrappers:

    constexpr NaNPropagComplex& operator*=(const T& x)
    {
        (std::complex<T>&)(*this) *= x;
        return *this;
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

    constexpr NaNPropagComplex& operator/=(const T& x)
    {
        (std::complex<T>&)(*this) /= x;
        return *this;
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
};

template <typename T>
struct real_type_traits<NaNPropagComplex<T>> {
    using type = T;
};

template <typename T>
struct complex_type_traits<NaNPropagComplex<T>> {
    using type = NaNPropagComplex<T>;
};

template <typename T1, typename T2>
struct scalar_type_traits<NaNPropagComplex<T1>, T2> {
    using type = NaNPropagComplex<typename std::common_type<T1, T2>::type>;
};

template <typename T1, typename T2>
struct scalar_type_traits<T1, NaNPropagComplex<T2>> {
    using type = NaNPropagComplex<typename std::common_type<T1, T2>::type>;
};

template <typename T1, typename T2>
struct scalar_type_traits<NaNPropagComplex<T1>, NaNPropagComplex<T2>> {
    using type = NaNPropagComplex<typename std::common_type<T1, T2>::type>;
};

template <typename T>
inline T abs(const NaNPropagComplex<T>& x)
{
    return tlapack::abs((std::complex<T>)x);
}

}  // namespace tlapack

#endif  // TLAPACK_TEST_COMPLEX_HPP