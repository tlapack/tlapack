/// @file mpreal.hpp
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

#include <complex>

namespace tlapack {

// Forward declarations
template <typename T>
T abs(const T& x);
template <typename T>
bool isnan(const std::complex<T>& x);
template <typename T>
bool isinf(const std::complex<T>& x);

/// Absolute value
template <>
inline mpfr::mpreal abs(const mpfr::mpreal& x)
{
    return mpfr::abs(x);
}

/// 2-norm absolute value, sqrt( |Re(x)|^2 + |Im(x)|^2 )
///
/// Note that std::abs< std::complex > does not overflow or underflow at
/// intermediate stages of the computation.
/// @see https://en.cppreference.com/w/cpp/numeric/complex/abs
/// but it may not propagate NaNs.
///
/// Also, std::abs< mpfr::mpreal > may not propagate Infs.
///
inline mpfr::mpreal abs(const std::complex<mpfr::mpreal>& x)
{
    if (isnan(x))
        return std::numeric_limits<mpfr::mpreal>::quiet_NaN();
    else if (isinf(x))
        return std::numeric_limits<mpfr::mpreal>::infinity();
    else
        return std::abs(x);
}

// Argument-dependent lookup (ADL) will include the remaining functions,
// e.g., mpfr::sin, mpfr::cos.
// Including them here may cause ambiguous call of overloaded function.
// See: https://en.cppreference.com/w/cpp/language/adl

#ifdef MPREAL_HAVE_DYNAMIC_STD_NUMERIC_LIMITS

// Forward declaration
template <typename real_t>
const int digits();

/// Specialization for the mpfr::mpreal datatype
template <>
inline const int digits<mpfr::mpreal>()
{
    return std::numeric_limits<mpfr::mpreal>::digits();
}
#endif

}  // namespace tlapack

#endif  // TLAPACK_MPREAL_HH
