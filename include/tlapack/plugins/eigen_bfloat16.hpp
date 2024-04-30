/// @file eigen_bfloat16.hpp Eigen::bfloat16 compatibility with
/// tlapack::concepts::Real
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_EIGEN_BFLOAT16_HH
#define TLAPACK_EIGEN_BFLOAT16_HH

#include <Eigen/Core>

#include "tlapack/base/types.hpp"

namespace tlapack {

namespace traits {
    // Eigen::bfloat16 is a real type that satisfies tlapack::concepts::Real
    template <>
    struct real_type_traits<Eigen::bfloat16, int> {
        using type = Eigen::bfloat16;
        constexpr static bool is_real = true;
    };
    // The complex type of Eigen::bfloat16 is std::complex<Eigen::bfloat16>
    template <>
    struct complex_type_traits<Eigen::bfloat16, int> {
        using type = std::complex<Eigen::bfloat16>;
        constexpr static bool is_complex = false;
    };
}  // namespace traits

inline Eigen::bfloat16 pow(int base, const Eigen::bfloat16& exp)
{
    return Eigen::bfloat16_impl::pow(Eigen::bfloat16(float(base)), exp);
}

// // Reimplementation of std::sqrt for Eigen::half. See the discussion at
// // https://github.com/gcc-mirror/gcc/pull/84
// inline std::complex<Eigen::half> sqrt(const std::complex<Eigen::half>& z)
// {
//     const Eigen::half x = real(z);
//     const Eigen::half y = imag(z);
//     const Eigen::half zero(0);
//     const Eigen::half two(2);
//     const Eigen::half half(0.5);

//     if (isnan(x) || isnan(y))
//         return std::numeric_limits<Eigen::half>::quiet_NaN();
//     else if (isinf(x) || isinf(y))
//         return std::numeric_limits<Eigen::half>::infinity();
//     else if (x == zero) {
//         Eigen::half t = sqrt(half * abs(y));
//         return std::complex<Eigen::half>(t, (y < zero) ? -t : t);
//     }
//     else {
//         Eigen::half t = sqrt(two * (std::abs(z) + abs(x)));
//         Eigen::half u = half * t;
//         return (x > zero)
//                    ? std::complex<Eigen::half>(u, y / t)
//                    : std::complex<Eigen::half>(abs(y) / t, (y < zero) ? -u :
//                    u);
//     }
// }

}  // namespace tlapack

inline std::istream& operator>>(std::istream& is, Eigen::bfloat16& x)
{
    float f;
    is >> f;
    x = Eigen::bfloat16(f);
    return is;
}

#endif  // TLAPACK_EIGEN_HALF_HH
