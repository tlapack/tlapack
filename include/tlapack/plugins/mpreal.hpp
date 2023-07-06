/// @file mpreal.hpp mpfr::mpreal compatibility with tlapack::concepts::Real
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MPREAL_HH
#define TLAPACK_MPREAL_HH

#include <mpreal.h>

#include "tlapack/base/types.hpp"

namespace tlapack {

namespace traits {
    // mpfr::mpreal is a real type that satisfies tlapack::concepts::Real
    template <>
    struct real_type_traits<mpfr::mpreal, int> {
        using type = mpfr::mpreal;
        constexpr static bool is_real = true;
    };
    // The complex type of mpfr::mpreal is std::complex<mpfr::mpreal>
    template <>
    struct complex_type_traits<mpfr::mpreal, int> {
        using type = std::complex<mpfr::mpreal>;
        constexpr static bool is_complex = false;
    };
}  // namespace traits

// Forward declarations
template <typename T>
inline T abs(const T& x);

template <>
inline mpfr::mpreal abs(const mpfr::mpreal& x)
{
    return mpfr::abs(x);
}

// Argument-dependent lookup (ADL) will include the remaining functions,
// e.g., mpfr::sin, mpfr::cos.
// Including them here may cause ambiguous call of overloaded function.
// See: https://en.cppreference.com/w/cpp/language/adl

#ifdef MPREAL_HAVE_DYNAMIC_STD_NUMERIC_LIMITS

// Forward declaration
template <typename real_t>
inline int digits();

// Specialization for the mpfr::mpreal datatype
template <>
inline int digits<mpfr::mpreal>()
{
    return std::numeric_limits<mpfr::mpreal>::digits();
}

#endif

}  // namespace tlapack

#endif  // TLAPACK_MPREAL_HH
