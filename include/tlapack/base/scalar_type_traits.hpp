/// @file scalar_type_traits.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_SCALAR_TRAITS_HH
#define TLAPACK_SCALAR_TRAITS_HH

#include <complex>
#include <type_traits>

namespace tlapack {

// C++ standard utils:
using std::enable_if_t;

// -----------------------------------------------------------------------------
// for any combination of types, determine associated real, scalar,
// and complex types.
//
// real_type< float >                               is float
// real_type< float, double, complex<float> >       is double
//
// scalar_type< float >                             is float
// scalar_type< float, complex<float> >             is complex<float>
// scalar_type< float, double, complex<float> >     is complex<double>
//
// complex_type< float >                            is complex<float>
// complex_type< float, double >                    is complex<double>
// complex_type< float, double, complex<float> >    is complex<double>
//
// Adds promotion of complex types based on the common type of the associated
// real types. This fixes various cases:
//
// std::std::common_type_t< double, complex<float> > is complex<float>  (wrong)
//        scalar_type< double, complex<float> > is complex<double> (right)
//
// std::std::common_type_t< int, complex<long> > is not defined (compile error)
//        scalar_type< int, complex<long> > is complex<long> (right)

/** Type traits in the library.
 *
 * @note For better readability, prefer using the aliases for the traits, if
 * available. For example, use @c tlapack::real_type<T> instead of
 * @c tlapack::traits::real_type_traits<T>::type.
 */
namespace traits {
    /** Real type traits for the list of types @c Types.
     *
     * The deduced common real type is defined on
     * @c real_type_traits<Types...>::type. For a single type @c T, the trait
     * also defines a static boolean @c is_real that is true if @c T is a real
     * type. Use the aliases tlapack::real_type and tlapack::is_real instead.
     *
     * Rules when a single type @c T is given:
     *
     * - If @c T is a non-const non-complex std arithmetic type, then the
     * deduced type is the decayed type of @c T.
     * - If @c T is a std complex type, then the deduced type is the decayed
     * type of the real part of @c T.
     * - If @c T is a const type, then the deduced type is the same as the one
     * of the non-const version.
     * - If @c T is a pointer or reference, then the deduced type is void. This
     * is to avoid accidental promotion of pointers and references.
     *
     * For two or more types, use @c std::common_type_t to deduce the common
     * type using recursive calls.
     *
     * @note This trait can be specialized for a given type (or list of types)
     * to override the default behavior.
     *
     * @note this trait can be specialized for matrices and vectors. This is
     * used in routines that need to create matrices (and/or vectors) with real
     * entries from a given matrix (and/or vector) type whose entries can be
     * complex.
     *
     * @tparam Types A list of types
     */
    template <typename... Types>
    struct real_type_traits;

    // Traits for one non-const non-complex std arithmetic type
    template <typename T>
    struct real_type_traits<
        T,
        enable_if_t<std::is_arithmetic_v<T> && !std::is_const_v<T>, int>> {
        using type = typename std::decay<T>::type;
        constexpr static bool is_real = true;
    };

    // Traits for a std complex type (strip complex)
    template <typename T>
    struct real_type_traits<std::complex<T>, int> {
        using type = typename real_type_traits<T, int>::type;
        constexpr static bool is_real = false;
    };

    // Traits for a const type (strip const)
    template <typename T>
    struct real_type_traits<const T, int> : public real_type_traits<T, int> {};

    // Pointers and references don't have a real type
    template <typename T>
    struct real_type_traits<
        T,
        enable_if_t<std::is_pointer_v<T> || std::is_reference_v<T>, int>> {
        using type = void;
        constexpr static bool is_real = false;
    };

    // Deduction for two or more types
    template <typename T1, typename T2, typename... Types>
    struct real_type_traits<T1, T2, Types...> {
        using type =
            std::common_type_t<typename real_type_traits<T1, int>::type,
                               typename real_type_traits<T2, Types...>::type>;
    };
}  // namespace traits

/// The common real type of the list of types
template <typename... Types>
using real_type = typename traits::real_type_traits<Types..., int>::type;

/// True if T is a real scalar type
template <typename T>
constexpr bool is_real = traits::real_type_traits<T, int>::is_real;

namespace traits {
    /** Complex type traits for the list of types @c Types.
     *
     * The deduced common complex type is defined on
     * @c complex_type_traits<Types...>::type. For a single type @c T, the trait
     * also defines a static boolean @c is_complex that is true if @c T is a
     * complex type. Use the aliases tlapack::complex_type and
     * tlapack::is_complex instead.
     *
     * Rules when a single type @c T is given:
     *
     * - If @c T is a non-const non-complex std arithmetic type or a std complex
     * type, then the deduced type is the std complex type of @c
     * tlapack::real_type<T>.
     * - If @c T is a const type, then the deduced type is the same as the one
     * of the non-const version.
     * - If @c T is a pointer or reference, then the deduced type is void. This
     * is to avoid accidental promotion of pointers and references.
     *
     * For two or more types, the deduced type is the complex type of
     * @c tlapack::real_type_traits<Types...>::type.
     *
     * @note This trait can be specialized for a given type (or list of types)
     * to override the default behavior.
     *
     * @note this trait can be specialized for matrices and vectors. This is
     * used in routines that need to create matrices (and/or vectors) with
     * complex entries from a given matrix (and/or vector) type whose entries
     * can be real.
     *
     * @tparam Types A list of types
     */
    template <typename... Types>
    struct complex_type_traits;

    // Traits for one non-const non-complex std arithmetic type
    template <typename T>
    struct complex_type_traits<
        T,
        enable_if_t<std::is_arithmetic_v<T> && !std::is_const_v<T>, int>> {
        using type = std::complex<real_type<T>>;
        constexpr static bool is_complex = false;
    };

    // Traits for a std complex type
    template <typename T>
    struct complex_type_traits<std::complex<T>, int> {
        using type = std::complex<real_type<T>>;
        constexpr static bool is_complex = true;
    };

    // Traits for a const type (strip const)
    template <typename T>
    struct complex_type_traits<const T, int>
        : public complex_type_traits<T, int> {};

    // Pointers and references don't have a complex type
    template <typename T>
    struct complex_type_traits<
        T,
        enable_if_t<std::is_pointer_v<T> || std::is_reference_v<T>, int>> {
        using type = void;
        constexpr static bool is_complex = false;
    };

    // for two or more types
    template <typename T1, typename T2, typename... Types>
    struct complex_type_traits<T1, T2, Types...> {
        using type = typename complex_type_traits<
            typename real_type_traits<T1, T2, Types...>::type,
            int>::type;
    };
}  // namespace traits

/// The common complex type of the list of types
template <typename... Types>
using complex_type = typename traits::complex_type_traits<Types..., int>::type;

/// True if T is a complex scalar type
template <typename T>
constexpr bool is_complex = traits::complex_type_traits<T, int>::is_complex;

namespace traits {
    /** Common scalar type deduced from the list of types @c Types.
     *
     * The deduced common scalar type is defined on
     * @c scalar_type_traits<Types...>::type. Use the alias tlapack::scalar_type
     * to get the deduced type.
     *
     * Rules:
     *
     * - For one type, the deduced type is the same the type deduced for the
     * pair of types @c T and @c T.
     * - For two types, if one of them is a complex type, then the deduced type
     * is the same type deduced by @c tlapack::complex_type.
     * - For two types, if both are real types, then the deduced type is the
     * same type deduced by @c tlapack::real_type.
     * - For three or more types, use a recursive call to deduce the common
     * scalar type of the first two types, and then recursively deduce the
     * common scalar type of the result and the remaining types.
     *
     * @tparam Types
     */
    template <typename... Types>
    struct scalar_type_traits;

    // for one type
    template <typename T>
    struct scalar_type_traits<T, int> : scalar_type_traits<T, T, int> {};

    // for two types, one is complex
    template <typename T1, typename T2>
    struct scalar_type_traits<
        T1,
        T2,
        enable_if_t<is_complex<T1> || is_complex<T2>, int>> {
        using type = complex_type<T1, T2>;
    };

    // for two types, neither is complex
    template <typename T1, typename T2>
    struct scalar_type_traits<T1,
                              T2,
                              enable_if_t<is_real<T1> && is_real<T2>, int>> {
        using type = real_type<T1, T2>;
    };

    // for three or more types
    template <typename T1, typename T2, typename... Types>
    struct scalar_type_traits<T1, T2, Types...> {
        using type = typename scalar_type_traits<
            typename scalar_type_traits<T1, T2, int>::type,
            Types...>::type;
    };
}  // namespace traits

/// The common scalar type of the list of types
template <typename... Types>
using scalar_type = typename traits::scalar_type_traits<Types..., int>::type;

}  // namespace tlapack

#endif  // TLAPACK_SCALAR_TRAITS_HH
