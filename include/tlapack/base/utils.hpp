/// @file utils.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
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

// -----------------------------------------------------------------------------
// From std C++
using std::ceil;
using std::enable_if_t;
using std::floor;
using std::is_same_v;
using std::isinf;
using std::isnan;
using std::max;
using std::min;
using std::pair;
using std::pow;
using std::sqrt;

//------------------------------------------------------------------------------

template <typename T, enable_if_t<is_real<T>, int> = 0>
inline constexpr real_type<T> real(const T& x)
{
    return x;
}

template <typename T, enable_if_t<is_real<T>, int> = 0>
inline constexpr real_type<T> imag(const T& x)
{
    return real_type<T>(0);
}

/** Extend conj to real datatypes.
 *
 * Usage:
 *
 *     using tlapack::conj;
 *     T x = ...
 *     T y = conj( x );
 *
 * @param[in] x Real number
 * @return x
 *
 * @note C++11 to C++17 returns complex<real_t> instead of real_t. @see
 * std::conj
 */
template <typename T, enable_if_t<is_real<T>, int> = 0>
inline constexpr T conj(const T& x)
{
    return x;
}

// -----------------------------------------------------------------------------
/// Type-safe sgn function
/// @see Source: https://stackoverflow.com/a/4609795/5253097
///
template <typename real_t>
inline int sgn(const real_t& val)
{
    return (real_t(0) < val) - (val < real_t(0));
}

// -----------------------------------------------------------------------------
/// isinf for complex numbers
template <typename real_t>
inline bool isinf(const std::complex<real_t>& x)
{
    return isinf(real(x)) || isinf(imag(x));
}

namespace internal {
    template <class T, typename = int>
    struct has_operator_parenthesis_with_2_indexes : std::false_type {};

    template <class T>
    struct has_operator_parenthesis_with_2_indexes<
        T,
        enable_if_t<!is_same_v<decltype(std::declval<T>()(0, 0)), void>, int>>
        : std::true_type {};

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

/**
 * Returns true if and only if A has an infinite entry.
 *
 * @tparam uplo_t Type of access inside the algorithm.
 *      Either Uplo or any type that implements
 *          operator Uplo().
 *
 * @param[in] uplo Determines the entries of A that will be checked.
 *      The following access types are allowed:
 *          Uplo::General,
 *          Uplo::UpperHessenberg,
 *          Uplo::LowerHessenberg,
 *          Uplo::Upper,
 *          Uplo::Lower,
 *          Uplo::StrictUpper,
 *          Uplo::StrictLower.
 *
 * @param[in] A matrix.
 *
 * @return true if A has an infinite entry.
 * @return false if A has no infinite entry.
 */
template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t>
bool hasinf(uplo_t uplo, const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (uplo == Uplo::UpperHessenberg) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j + 2 : m); ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j + 1 : m); ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::StrictUpper) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j : m); ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::LowerHessenberg) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = ((j > 1) ? j - 1 : 0); i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::Lower) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j; i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::StrictLower) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 1; i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else  // if ( (Uplo) uplo == Uplo::General )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
}

/**
 * Returns true if and only if A has an infinite entry.
 *
 * Specific implementation for band access types.
 * @see tlapack::hasinf(uplo_t uplo, const matrix_t& A).
 */
template <TLAPACK_MATRIX matrix_t>
bool hasinf(BandAccess accessType, const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kl = accessType.lower_bandwidth;
    const idx_t ku = accessType.upper_bandwidth;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = ((j >= ku) ? (j - ku) : 0); i < min(m, j + kl + 1); ++i)
            if (isinf(A(i, j))) return true;
    return false;
}

/**
 * Returns true if and only if x has an infinite entry.
 *
 * @param[in] x vector.
 *
 * @return true if x has an infinite entry.
 * @return false if x has no infinite entry.
 */
template <TLAPACK_VECTOR vector_t>
bool hasinf(const vector_t& x)
{
    using idx_t = size_type<vector_t>;

    // constants
    const idx_t n = size(x);

    for (idx_t i = 0; i < n; ++i)
        if (isinf(x[i])) return true;
    return false;
}

// -----------------------------------------------------------------------------
/// isnan for complex numbers
template <typename real_t>
inline bool isnan(const std::complex<real_t>& x)
{
    return isnan(real(x)) || isnan(imag(x));
}

/**
 * Returns true if and only if A has an NaN entry.
 *
 * @tparam uplo_t Type of access inside the algorithm.
 *      Either Uplo or any type that implements
 *          operator Uplo().
 *
 * @param[in] uplo Determines the entries of A that will be checked.
 *      The following access types are allowed:
 *          Uplo::General,
 *          Uplo::UpperHessenberg,
 *          Uplo::LowerHessenberg,
 *          Uplo::Upper,
 *          Uplo::Lower,
 *          Uplo::StrictUpper,
 *          Uplo::StrictLower.
 *
 * @param[in] A matrix.
 *
 * @return true if A has an NaN entry.
 * @return false if A has no NaN entry.
 */
template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t>
bool hasnan(uplo_t uplo, const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (uplo == Uplo::UpperHessenberg) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j + 2 : m); ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j + 1 : m); ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::StrictUpper) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j : m); ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::LowerHessenberg) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = ((j > 1) ? j - 1 : 0); i < m; ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::Lower) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j; i < m; ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::StrictLower) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 1; i < m; ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else  // if ( (Uplo) uplo == Uplo::General )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
}

/**
 * Returns true if and only if A has an NaN entry.
 *
 * Specific implementation for band access types.
 * @see tlapack::hasnan(uplo_t uplo, const matrix_t& A).
 */
template <TLAPACK_MATRIX matrix_t>
bool hasnan(BandAccess accessType, const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kl = accessType.lower_bandwidth;
    const idx_t ku = accessType.upper_bandwidth;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = ((j >= ku) ? (j - ku) : 0); i < min(m, j + kl + 1); ++i)
            if (isnan(A(i, j))) return true;
    return false;
}

/**
 * Returns true if and only if x has an NaN entry.
 *
 * @param[in] x vector.
 *
 * @return true if x has an NaN entry.
 * @return false if x has no NaN entry.
 */
template <TLAPACK_VECTOR vector_t>
bool hasnan(const vector_t& x)
{
    using idx_t = size_type<vector_t>;

    // constants
    const idx_t n = size(x);

    for (idx_t i = 0; i < n; ++i)
        if (isnan(x[i])) return true;
    return false;
}

// -----------------------------------------------------------------------------
// Absolute value

/** 2-norm absolute value, sqrt( |Re(x)|^2 + |Im(x)|^2 )
 *
 * Note that std::abs< std::complex > does not overflow or underflow at
 * intermediate stages of the computation.
 * @see https://en.cppreference.com/w/cpp/numeric/complex/abs
 * but it may not propagate NaNs.
 *
 * Also, std::abs< mpfr::mpreal > may not propagate Infs.
 */
template <typename T>
inline T abs(const T& x);

inline float abs(float x) { return std::fabs(x); }
inline double abs(double x) { return std::fabs(x); }
inline long double abs(long double x) { return std::fabs(x); }

template <typename T>
inline T abs(const std::complex<T>& x)
{
    // If the default value of ErrorCheck::nan is true then check for NaNs
    if (ErrorCheck().nan && isnan(x))
        return std::numeric_limits<T>::quiet_NaN();
    // If the default value of ErrorCheck::inf is true then check for Infs
    else if (ErrorCheck().inf && isinf(x))
        return std::numeric_limits<T>::infinity();
    else
        return std::abs(x);
}

// -----------------------------------------------------------------------------
/// 1-norm absolute value, |Re(x)| + |Im(x)|
template <typename T>
real_type<T> abs1(const T& x)
{
    return abs(real(x)) + abs(imag(x));
}

// -----------------------------------------------------------------------------
/// Optimized BLAS

namespace internal {

    /// True if C1, C2, Cs... have all compatible layouts. False otherwise.
    template <class C1, class C2, class... Cs>
    constexpr bool has_compatible_layout = (has_compatible_layout<C1, C2> &&
                                            has_compatible_layout<C1, Cs...> &&
                                            has_compatible_layout<C2, Cs...>);

    /// True if C1 and C2 are matrices with same layout.
    /// Also true if C1 or C2 are strided vectors or scalars
    template <class C1, class C2>
    constexpr bool has_compatible_layout<C1, C2> =
        (!is_matrix<C1> && !is_vector<C1>) ||
        (!is_matrix<C2> && !is_vector<C2>) || (layout<C1> == Layout::Strided) ||
        (layout<C2> == Layout::Strided) || (layout<C1> == layout<C2>);

    /// True if pair C1 and C2 have compatible layouts.
    template <class C1, class T1, class C2, class T2>
    constexpr bool has_compatible_layout<pair<C1, T1>, pair<C2, T2>> =
        has_compatible_layout<C1, C2>;

    /**
     * @brief Trait to determine if a given list of data allows optimization
     * using a optimized BLAS library.
     *
     * @tparam Ts If the last class in this list is not an int, then the trait
     * is not defined.
     */
    template <class... Ts>
    struct allow_optblas_trait {
        static constexpr bool value =
            false;  ///< True if the list of types
                    ///< allows optimized BLAS library.
    };
}  // namespace internal

/// Alias for @c internal::allow_optblas_trait<,int>::value.
template <class... Ts>
constexpr bool allow_optblas = internal::allow_optblas_trait<Ts..., int>::value;

namespace internal {

    /// True if C is a row- or column-major matrix and the entry type can be
    /// used with optimized BLAS implementations.
    template <class C>
    struct allow_optblas_trait<
        C,
        enable_if_t<is_matrix<C> && !is_vector<C>, int>> {
        static constexpr bool value =
            allow_optblas<type_t<C>> &&
            (layout<C> == Layout::ColMajor || layout<C> == Layout::RowMajor);
    };

    /// True if C is a strided vector and the entry type can be used with
    /// optimized BLAS implementations.
    template <class C>
    struct allow_optblas_trait<C, enable_if_t<is_vector<C>, int>> {
        static constexpr bool value =
            allow_optblas<type_t<C>> &&
            (layout<C> == Layout::ColMajor || layout<C> == Layout::RowMajor ||
             layout<C> == Layout::Strided);
    };

    /** A pair of types <C,T> allows optimized BLAS if T allows optimized BLAS
     * and one of the followings happens:
     * 1. C is a matrix or vector that allows optimized BLAS and the entry type
     * is the same as T.
     * 2. C is not a matrix or vector, but is convertible to T.
     */
    template <class C, class T>
    struct allow_optblas_trait<pair<C, T>,
                               enable_if_t<is_matrix<C> || is_vector<C>, int>> {
        static constexpr bool value =
            allow_optblas<T> &&
            (allow_optblas<C> &&
             is_same_v<type_t<C>, typename std::decay<T>::type>);
    };
    template <class C, class T>
    struct allow_optblas_trait<
        pair<C, T>,
        enable_if_t<!is_matrix<C> && !is_vector<C>, int>> {
        static constexpr bool value =
            allow_optblas<T> && std::is_constructible<T, C>::value;
    };

    template <class C1, class T1, class C2, class T2, class... Ps>
    struct allow_optblas_trait<pair<C1, T1>, pair<C2, T2>, Ps...> {
        static constexpr bool value =
            allow_optblas_trait<pair<C1, T1>, int>::value &&
            allow_optblas_trait<pair<C2, T2>, Ps...>::value &&
            has_compatible_layout<C1, C2, Ps...>;
    };
}  // namespace internal

template <class T1, class... Ts>
using enable_if_allow_optblas_t = enable_if_t<(allow_optblas<T1, Ts...>), int>;

template <class T1, class... Ts>
using disable_if_allow_optblas_t =
    enable_if_t<(!allow_optblas<T1, Ts...>), int>;

#define TLAPACK_OPT_TYPE(T)                     \
    namespace internal {                        \
        template <>                             \
        struct allow_optblas_trait<T, int> {    \
            static constexpr bool value = true; \
        };                                      \
    }

/// Optimized types
#ifdef TLAPACK_USE_LAPACKPP
TLAPACK_OPT_TYPE(float)
TLAPACK_OPT_TYPE(double)
TLAPACK_OPT_TYPE(std::complex<float>)
TLAPACK_OPT_TYPE(std::complex<double>)
TLAPACK_OPT_TYPE(StrongZero)
#endif
#undef TLAPACK_OPT_TYPE

// -----------------------------------------------------------------------------
// Workspace:

/**
 * @brief Allocates workspace
 *
 * @param[out] v On exit, reference to allocated memory.
 * @param[in] lwork Number of bytes to allocate.
 *
 * @return Workspace referencing the allocated memory.
 */
inline Workspace alloc_workspace(VectorOfBytes& v, std::size_t lwork)
{
    v = VectorOfBytes(lwork);  // Allocates space in memory
    return Workspace(v.data(), v.size());
}

/**
 * @brief Allocates workspace
 *
 * @param[out] v        On exit, reference to allocated memory if needed.
 * @param[in] workinfo  Information about the amount of workspace required.
 * @param[in] opts_w    Workspace previously allocated.
 *
 * @return Workspace referencing either:
 *      1. new allocated memory, if opts_w.size() <= 0.
 *      2. previously allocated memory, if opts_w.size() >= lwork.
 */
inline Workspace alloc_workspace(VectorOfBytes& v,
                                 const WorkInfo& workinfo,
                                 const Workspace& opts_w = {})
{
    if (opts_w.size() <= 0) {
        return alloc_workspace(v, workinfo.size());
    }
    else {
        tlapack_check(
            (opts_w.isContiguous() && (opts_w.size() >= workinfo.size())) ||
            (opts_w.getM() >= workinfo.m && opts_w.getN() >= workinfo.n));

        return Workspace(opts_w);
    }
}

namespace internal {
    /** Chooses between a preferrable type `work_type` and a default type
     * `work_default`
     *
     * @c deduce_work<>::type = work_default only if deduce_work is void.
     *
     * @tparam work_type    Preferrable workspace type
     * @tparam work_default Default workspace type
     */
    template <class work_type, class work_default>
    struct deduce_work {
        using type = work_type;
    };
    template <class work_default>
    struct deduce_work<void, work_default> {
        using type = work_default;
    };
}  // namespace internal

/// Alias for @c deduce_work<>::type
template <class work_type, class work_default>
using deduce_work_t =
    typename internal::deduce_work<work_type, work_default>::type;

/**
 * @brief Options structure with a Workspace attribute
 *
 * @tparam work_t Give specialized data type to the workspaces.
 *      Behavior defined by each implementation using this option.
 */
template <class... work_t>
struct WorkspaceOpts {
    Workspace work;  ///< Workspace object

    // Constructors:

    inline constexpr WorkspaceOpts(Workspace&& w = {}) : work(w) {}

    inline constexpr WorkspaceOpts(const Workspace& w) : work(w) {}

    template <TLAPACK_LEGACY_ARRAY matrix_t>
    inline constexpr WorkspaceOpts(const matrix_t& A) : work(legacy_matrix(A))
    {}
};

}  // namespace tlapack

#endif  // TLAPACK_UTILS_HH
