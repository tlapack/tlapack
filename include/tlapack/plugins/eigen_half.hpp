/// @file eigen_half.hpp Eigen::half compatibility with tlapack::concepts::Real
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_EIGEN_HALF_HH
#define TLAPACK_EIGEN_HALF_HH

#include <Eigen/Core>

#include "tlapack/base/types.hpp"

namespace tlapack {



namespace traits {
    // Eigen::half is a real type that satisfies tlapack::concepts::Real
    template <>
    struct real_type_traits<Eigen::half, int> {
        using type = Eigen::half;
        constexpr static bool is_real = true;
    };
    // The complex type of Eigen::half is std::complex<Eigen::half>
    template <>
    struct complex_type_traits<Eigen::half, int> {
        using type = std::complex<Eigen::half>;
        constexpr static bool is_complex = false;
    };
}  // namespace traits

inline Eigen::half pow(int base, const Eigen::half& exp)
{
    return Eigen::half_impl::pow(Eigen::half(base), exp);
}

// Reimplementation of std::sqrt for Eigen::half. See the discussion at
// https://github.com/gcc-mirror/gcc/pull/84
inline std::complex<Eigen::half> sqrt(const std::complex<Eigen::half>& z)
{
    const Eigen::half x = real(z);
    const Eigen::half y = imag(z);
    const Eigen::half zero(0);
    const Eigen::half two(2);
    const Eigen::half half(0.5);

    if (isnan(x) || isnan(y))
        return std::numeric_limits<Eigen::half>::quiet_NaN();
    else if (isinf(x) || isinf(y))
        return std::numeric_limits<Eigen::half>::infinity();
    else if (x == zero) {
        Eigen::half t = sqrt(half * abs(y));
        return std::complex<Eigen::half>(t, (y < zero) ? -t : t);
    }
    else {
        Eigen::half t = sqrt(two * (std::abs(z) + abs(x)));
        Eigen::half u = half * t;
        return (x > zero)
                   ? std::complex<Eigen::half>(u, y / t)
                   : std::complex<Eigen::half>(abs(y) / t, (y < zero) ? -u : u);
    }
}

}  // namespace tlapack

inline std::istream& operator>>(std::istream& is, Eigen::half& x)
{
    float f;
    is >> f;
    x = Eigen::half(f);
    return is;
}

#endif  // TLAPACK_EIGEN_HALF_HH
