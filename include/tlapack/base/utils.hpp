/// @file utils.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UTILS_HH
#define TLAPACK_UTILS_HH

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/concepts.hpp"
#include "tlapack/base/exceptionHandling.hpp"
#include "tlapack/base/types.hpp"
#include "tlapack/base/workspace.hpp"

#ifdef TLAPACK_USE_LAPACKPP
    #include "lapack.hh"  // from LAPACK++
#endif

namespace tlapack {

// C++ standard utils:
using std::enable_if_t;
using std::is_same_v;

// C++ standard math functions:
using std::abs;
using std::ceil;
using std::floor;
using std::isinf;
using std::isnan;
using std::log2;
using std::max;
using std::min;
using std::pow;  // We only use pow(int, T), see below in the concept Real.
using std::sqrt;

// C++ standard types:
using std::pair;

// -----------------------------------------------------------------------------
// Utility function for squaring a number to avoid using pow for everything
template <typename T>
constexpr T square(const T& x)
{
    return x * x;
}

//------------------------------------------------------------------------------
// Extends std::real(), std::imag(), std::conj() to real numbers

/** Extends std::real() to real datatypes.
 *
 * @param[in] x Real number
 * @return x
 *
 * @note std::real() is already defined for real numbers in C++. However,
 * we want a more general definition that works for any real type, not only
 * for the built-in types.
 */
template <typename T, enable_if_t<is_real<T>, int> = 0>
constexpr real_type<T> real(const T& x) noexcept
{
    return real_type<T>(x);
}

/** Extends std::imag() to real datatypes.
 *
 * @param[in] x Real number
 * @return 0
 *
 * @note std::imag() is already defined for real numbers in C++. However,
 * we want a more general definition that works for any real type, not only
 * for the built-in types.
 */
template <typename T, enable_if_t<is_real<T>, int> = 0>
constexpr real_type<T> imag(const T& x) noexcept
{
    return real_type<T>(0);
}

/** Extends std::conj() to real datatypes.
 *
 * @param[in] x Real number
 * @return x
 *
 * @note std::conj() is already defined for real numbers in C++. However, the
 * return type is complex<real_t> instead of real_t.
 */
template <typename T, enable_if_t<is_real<T>, int> = 0>
constexpr T conj(const T& x) noexcept
{
    return x;
}

// -----------------------------------------------------------------------------
/// Type-safe sgn function
/// @see Source: https://stackoverflow.com/a/4609795/5253097
template <typename T, enable_if_t<is_real<T>, int> = 0>
constexpr int sgn(const T& val)
{
    return (T(0) < val) - (val < T(0));
}

// -----------------------------------------------------------------------------
/// Extends std::isinf() to complex numbers
template <typename T, enable_if_t<is_complex<T>, int> = 0>
constexpr bool isinf(const T& x) noexcept
{
    return isinf(real(x)) || isinf(imag(x));
}

// -----------------------------------------------------------------------------
/// Extends std::isnan() to complex numbers
template <typename T, enable_if_t<is_complex<T>, int> = 0>
constexpr bool isnan(const T& x) noexcept
{
    return isnan(real(x)) || isnan(imag(x));
}

// -----------------------------------------------------------------------------
/// 1-norm absolute value, |Re(x)| + |Im(x)|
template <typename T>
constexpr real_type<T> abs1(const T& x)
{
    return abs(real(x)) + abs(imag(x));
}

// -----------------------------------------------------------------------------
// Internal traits: is_matrix and is_vector

namespace traits {
    namespace internal {
        template <class T, typename = int>
        struct has_operator_parenthesis_with_2_indexes : std::false_type {};

        template <class T>
        struct has_operator_parenthesis_with_2_indexes<
            T,
            enable_if_t<!is_same_v<decltype(std::declval<T>()(0, 0)), void>,
                        int>> : std::true_type {};

        template <class T, typename = int>
        struct has_operator_brackets_with_1_index : std::false_type {};

        template <class T>
        struct has_operator_brackets_with_1_index<
            T,
            enable_if_t<!is_same_v<decltype(std::declval<T>()[0]), void>, int>>
            : std::true_type {};

        template <class T>
        constexpr bool is_matrix =
            has_operator_parenthesis_with_2_indexes<T>::value;

        template <class T>
        constexpr bool is_vector = has_operator_brackets_with_1_index<T>::value;
    }  // namespace internal
}  // namespace traits

// -----------------------------------------------------------------------------
// Specialization of traits::entry_type_trait and traits::size_type_trait
// matrices and vectors

namespace traits {
    template <class matrix_t>
    struct entry_type_trait<matrix_t,
                            enable_if_t<internal::is_matrix<matrix_t> &&
                                            !internal::is_vector<matrix_t>,
                                        int>> {
        using type = std::decay_t<decltype(
            ((const matrix_t)std::declval<matrix_t>())(0, 0))>;
    };

    template <class vector_t>
    struct entry_type_trait<vector_t,
                            enable_if_t<internal::is_vector<vector_t>, int>> {
        using type = std::decay_t<decltype(
            ((const vector_t)std::declval<vector_t>())[0])>;
    };

    template <class matrix_t>
    struct size_type_trait<matrix_t,
                           enable_if_t<internal::is_matrix<matrix_t> &&
                                           !internal::is_vector<matrix_t>,
                                       int>> {
        using type = std::decay_t<decltype(nrows(std::declval<matrix_t>()))>;
    };

    template <class vector_t>
    struct size_type_trait<vector_t,
                           enable_if_t<internal::is_vector<vector_t>, int>> {
        using type = std::decay_t<decltype(size(std::declval<vector_t>()))>;
    };
}  // namespace traits

// -----------------------------------------------------------------------------
// Optimized BLAS traits
//
// allow_optblas<>, enable_if_allow_optblas_t<>, disable_if_allow_optblas_t<>

namespace traits {

    namespace internal {
        /// True if C1, C2, Cs... have all compatible layouts. False otherwise.
        template <class C1, class C2, class... Cs>
        constexpr bool has_compatible_layout =
            (has_compatible_layout<C1, C2> &&
             has_compatible_layout<C1, Cs...> &&
             has_compatible_layout<C2, Cs...>);

        // True if C1 and C2 are matrices with same layout.
        // Also true if C1 or C2 are strided vectors or scalars
        template <class C1, class C2>
        constexpr bool has_compatible_layout<C1, C2> =
            (!is_matrix<C1> && !is_vector<C1>) ||
            (!is_matrix<C2> && !is_vector<C2>) ||
            (layout<C1> == Layout::Strided) ||
            (layout<C2> == Layout::Strided) || (layout<C1> == layout<C2>);

        // True if C1 and C2 have compatible layouts.
        template <class C1, class T1, class C2, class T2>
        constexpr bool has_compatible_layout<pair<C1, T1>, pair<C2, T2>> =
            has_compatible_layout<C1, C2>;
    }  // namespace internal

    /// Trait to determine if the list @c Types allows optimization.
    template <class... Types>
    struct allow_optblas_trait : std::false_type {};

    // True if C is a row- or column-major matrix and the entry type can be
    // used with optimized BLAS implementations.
    template <class C>
    struct allow_optblas_trait<
        C,
        enable_if_t<internal::is_matrix<C> && !internal::is_vector<C>, int>> {
        static constexpr bool value =
            allow_optblas_trait<type_t<C>, int>::value &&
            (layout<C> == Layout::ColMajor || layout<C> == Layout::RowMajor);
    };

    // True if C is a strided vector and the entry type can be used with
    // optimized BLAS implementations.
    template <class C>
    struct allow_optblas_trait<C, enable_if_t<internal::is_vector<C>, int>> {
        static constexpr bool value =
            allow_optblas_trait<type_t<C>, int>::value &&
            (layout<C> == Layout::ColMajor || layout<C> == Layout::RowMajor ||
             layout<C> == Layout::Strided);
    };

    // A pair of types <C,T> allows optimized BLAS if T allows optimized BLAS
    // and one of the followings happens:
    // 1. C is a matrix or vector that allows optimized BLAS and the entry type
    // is the same as T.
    // 2. C is not a matrix or vector, but is convertible to T.
    template <class C, class T>
    struct allow_optblas_trait<
        pair<C, T>,
        enable_if_t<internal::is_matrix<C> || internal::is_vector<C>, int>> {
        static constexpr bool value =
            allow_optblas_trait<T, int>::value &&
            (allow_optblas_trait<C, int>::value &&
             is_same_v<type_t<C>, typename std::decay<T>::type>);
    };
    template <class C, class T>
    struct allow_optblas_trait<
        pair<C, T>,
        enable_if_t<!internal::is_matrix<C> && !internal::is_vector<C>, int>> {
        static constexpr bool value = allow_optblas_trait<T, int>::value &&
                                      std::is_constructible<T, C>::value;
    };

    template <class C1, class T1, class C2, class T2, class... Ps>
    struct allow_optblas_trait<pair<C1, T1>, pair<C2, T2>, Ps...> {
        static constexpr bool value =
            allow_optblas_trait<pair<C1, T1>, int>::value &&
            allow_optblas_trait<pair<C2, T2>, Ps...>::value &&
            internal::has_compatible_layout<C1, C2, Ps...>;
    };
}  // namespace traits

/// True if the list of types allows optimized BLAS library.
template <class... Ts>
constexpr bool allow_optblas = traits::allow_optblas_trait<Ts..., int>::value;

/// Enable if the list of types allows optimized BLAS library.
template <class T1, class... Ts>
using enable_if_allow_optblas_t = enable_if_t<(allow_optblas<T1, Ts...>), int>;

/// Disable if the list of types allows optimized BLAS library.
template <class T1, class... Ts>
using disable_if_allow_optblas_t =
    enable_if_t<(!allow_optblas<T1, Ts...>), int>;

#ifdef TLAPACK_USE_LAPACKPP
namespace traits {
    template <>
    struct allow_optblas_trait<float, int> : std::true_type {};
    template <>
    struct allow_optblas_trait<double, int> : std::true_type {};
    template <>
    struct allow_optblas_trait<std::complex<float>, int> : std::true_type {};
    template <>
    struct allow_optblas_trait<std::complex<double>, int> : std::true_type {};
    template <>
    struct allow_optblas_trait<StrongZero, int> : std::true_type {};
}  // namespace traits
#endif  // TLAPACK_USE_LAPACKPP

}  // namespace tlapack

#endif  // TLAPACK_UTILS_HH
