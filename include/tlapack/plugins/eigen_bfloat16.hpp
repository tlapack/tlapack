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

}  // namespace tlapack

inline std::istream& operator>>(std::istream& is, Eigen::bfloat16& x)
{
    float f;
    is >> f;
    x = Eigen::bfloat16(f);
    return is;
}

#endif  // TLAPACK_EIGEN_HALF_HH
