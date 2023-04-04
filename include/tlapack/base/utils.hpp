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
#include "tlapack/base/exceptionHandling.hpp"
#include "tlapack/base/types.hpp"
#include "tlapack/base/workspace.hpp"

#ifdef USE_LAPACKPP_WRAPPERS
    #include "lapack.hh"  // from LAPACK++
#endif

namespace tlapack {

// -----------------------------------------------------------------------------
// From std C++
using std::atan;
using std::ceil;
using std::cos;
using std::exp;
using std::floor;
using std::isinf;
using std::isnan;
using std::pair;
using std::pow;
using std::sin;
using std::sqrt;

template <typename idx_t>
using range = pair<idx_t, idx_t>;
using std::enable_if_t;

// -----------------------------------------------------------------------------
// is_same_v, is_convertible_v are defined in C++17; here's a C++11 definition
#if __cplusplus >= 201703L
using std::is_convertible_v;
using std::is_same_v;
#else
template <class T, class U>
constexpr bool is_convertible_v = std::is_convertible<T, U>::value;
template <class T, class U>
constexpr bool is_same_v = std::is_same<T, U>::value;
#endif

//------------------------------------------------------------------------------
/// True if T is complex_type<T>
template <typename T>
struct is_complex {
    static constexpr bool value = is_same_v<complex_type<T>, T>;
};

template <typename T, enable_if_t<!is_complex<T>::value, int> = 0>
inline constexpr real_type<T> real(const T& x)
{
    return x;
}

template <typename T, enable_if_t<!is_complex<T>::value, int> = 0>
inline constexpr real_type<T> imag(const T& x)
{
    return real_type<T>(0);
}

/** Extend conj to real datatypes.
 *
 * Usage:
 *
 *     using tlapack::conj;
 *     scalar_t x = ...
 *     scalar_t y = conj( x );
 *
 * @param[in] x Real number
 * @return x
 *
 * @note C++11 to C++17 returns complex<real_t> instead of real_t. @see
 * std::conj
 */
template <typename real_t, enable_if_t<!is_complex<real_t>::value, int> = 0>
inline constexpr real_t conj(const real_t& x)
{
    return x;
}

// -----------------------------------------------------------------------------
// max that works with different data types
// and any number of arguments: max( a, b, c, d )

// one argument
template <typename T>
inline T max(const T& x)
{
    return x;
}

// two arguments
template <typename T1, typename T2>
inline scalar_type<T1, T2> max(const T1& x, const T2& y)
{
    return (x >= y ? x : y);
}

// three or more arguments
template <typename T1, typename... Types>
inline scalar_type<T1, Types...> max(const T1& first, const Types&... args)
{
    return max(first, max(args...));
}

// -----------------------------------------------------------------------------
// min that works with different data types
// and any number of arguments: min( a, b, c, d )

// one argument
template <typename T>
inline T min(const T& x)
{
    return x;
}

// two arguments
template <typename T1, typename T2>
inline scalar_type<T1, T2> min(const T1& x, const T2& y)
{
    return (x <= y ? x : y);
}

// three or more arguments
template <typename T1, typename... Types>
inline scalar_type<T1, Types...> min(const T1& first, const Types&... args)
{
    return min(first, min(args...));
}

// -----------------------------------------------------------------------------
// Generate a scalar from real and imaginary parts.
// For real scalars, the imaginary part is ignored.

// For real scalar types.
template <typename real_t>
struct MakeScalarTraits {
    static inline real_t make(const real_t& re, const real_t& im) { return re; }
};

// For complex scalar types.
template <typename real_t>
struct MakeScalarTraits<std::complex<real_t>> {
    static inline std::complex<real_t> make(const real_t& re, const real_t& im)
    {
        return std::complex<real_t>(re, im);
    }
};

template <typename scalar_t>
inline scalar_t make_scalar(real_type<scalar_t> re, real_type<scalar_t> im = 0)
{
    return MakeScalarTraits<scalar_t>::make(re, im);
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

    template <class array_t, typename = int>
    struct has_operator_parenthesis_with_2_indexes : std::false_type {};

    template <class array_t>
    struct has_operator_parenthesis_with_2_indexes<
        array_t,
        enable_if_t<!is_same_v<decltype(std::declval<array_t>()(0, 0)), void>,
                    int>> : std::true_type {};

    template <class array_t, typename = int>
    struct has_operator_brackets_with_1_index : std::false_type {};

    template <class array_t>
    struct has_operator_brackets_with_1_index<
        array_t,
        enable_if_t<!is_same_v<decltype(std::declval<array_t>()[0]), void>,
                    int>> : std::true_type {};

}  // namespace internal

template <class T>
constexpr bool is_matrix =
    internal::has_operator_parenthesis_with_2_indexes<T>::value;

template <class T>
constexpr bool is_vector =
    !is_matrix<T> && internal::has_operator_brackets_with_1_index<T>::value;

template <class T>
constexpr bool is_scalar = !is_matrix<T> && !is_vector<T>;

namespace internal {

    /**
     * @brief Data type trait.
     *
     * The data type is defined on @c type_trait<array_t>::type.
     *
     * @tparam matrix_t Matrix class.
     */
    template <class matrix_t>
    struct type_trait<matrix_t, enable_if_t<is_matrix<matrix_t>, int>> {
        using type =
            typename std::decay<decltype(std::declval<matrix_t>()(0, 0))>::type;
    };

    /**
     * @brief Data type trait.
     *
     * The data type is defined on @c type_trait<array_t>::type.
     *
     * @tparam vector_t Vector class.
     */
    template <class vector_t>
    struct type_trait<vector_t, enable_if_t<is_vector<vector_t>, int>> {
        using type =
            typename std::decay<decltype(std::declval<vector_t>()[0])>::type;
    };

    /**
     * @brief Size type trait.
     *
     * The size type is defined on @c sizet_trait<array_t>::type.
     *
     * @tparam matrix_t Matrix class.
     */
    template <class matrix_t>
    struct sizet_trait<matrix_t, enable_if_t<is_matrix<matrix_t>, int>> {
        using type = typename std::decay<decltype(
            nrows(std::declval<matrix_t>()))>::type;
    };

    /**
     * @brief Size type trait.
     *
     * The size type is defined on @c sizet_trait<array_t>::type.
     *
     * @tparam vector_t Vector class.
     */
    template <class vector_t>
    struct sizet_trait<vector_t, enable_if_t<is_vector<vector_t>, int>> {
        using type =
            typename std::decay<decltype(size(std::declval<vector_t>()))>::type;
    };

}  // namespace internal

/**
 * Returns true if and only if A has an infinite entry.
 *
 * @tparam access_t Type of access inside the algorithm.
 *      Either MatrixAccessPolicy or any type that implements
 *          operator MatrixAccessPolicy().
 *
 * @param[in] accessType Determines the entries of A that will be checked.
 *      The following access types are allowed:
 *          MatrixAccessPolicy::Dense,
 *          MatrixAccessPolicy::UpperHessenberg,
 *          MatrixAccessPolicy::LowerHessenberg,
 *          MatrixAccessPolicy::UpperTriangle,
 *          MatrixAccessPolicy::LowerTriangle,
 *          MatrixAccessPolicy::StrictUpper,
 *          MatrixAccessPolicy::StrictLower.
 *
 * @param[in] A matrix.
 *
 * @return true if A has an infinite entry.
 * @return false if A has no infinite entry.
 */
template <class access_t, class matrix_t>
bool hasinf(access_t accessType, const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if ((MatrixAccessPolicy)accessType == MatrixAccessPolicy::UpperHessenberg) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j + 2 : m); ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if ((MatrixAccessPolicy)accessType ==
             MatrixAccessPolicy::UpperTriangle) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j + 1 : m); ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if ((MatrixAccessPolicy)accessType ==
             MatrixAccessPolicy::StrictUpper) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j : m); ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if ((MatrixAccessPolicy)accessType ==
             MatrixAccessPolicy::LowerHessenberg) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = ((j > 1) ? j - 1 : 0); i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if ((MatrixAccessPolicy)accessType ==
             MatrixAccessPolicy::LowerTriangle) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j; i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if ((MatrixAccessPolicy)accessType ==
             MatrixAccessPolicy::StrictLower) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 1; i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else  // if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::Dense )
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
 * @see hasinf( access_t accessType, const matrix_t& A ).
 */
template <class matrix_t>
bool hasinf(band_t accessType, const matrix_t& A)
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
template <class vector_t>
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
 * @tparam access_t Type of access inside the algorithm.
 *      Either MatrixAccessPolicy or any type that implements
 *          operator MatrixAccessPolicy().
 *
 * @param[in] accessType Determines the entries of A that will be checked.
 *      The following access types are allowed:
 *          MatrixAccessPolicy::Dense,
 *          MatrixAccessPolicy::UpperHessenberg,
 *          MatrixAccessPolicy::LowerHessenberg,
 *          MatrixAccessPolicy::UpperTriangle,
 *          MatrixAccessPolicy::LowerTriangle,
 *          MatrixAccessPolicy::StrictUpper,
 *          MatrixAccessPolicy::StrictLower.
 *
 * @param[in] A matrix.
 *
 * @return true if A has an NaN entry.
 * @return false if A has no NaN entry.
 */
template <class access_t, class matrix_t>
bool hasnan(access_t accessType, const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if ((MatrixAccessPolicy)accessType == MatrixAccessPolicy::UpperHessenberg) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j + 2 : m); ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else if ((MatrixAccessPolicy)accessType ==
             MatrixAccessPolicy::UpperTriangle) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j + 1 : m); ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else if ((MatrixAccessPolicy)accessType ==
             MatrixAccessPolicy::StrictUpper) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j : m); ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else if ((MatrixAccessPolicy)accessType ==
             MatrixAccessPolicy::LowerHessenberg) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = ((j > 1) ? j - 1 : 0); i < m; ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else if ((MatrixAccessPolicy)accessType ==
             MatrixAccessPolicy::LowerTriangle) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j; i < m; ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else if ((MatrixAccessPolicy)accessType ==
             MatrixAccessPolicy::StrictLower) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 1; i < m; ++i)
                if (isnan(A(i, j))) return true;
        return false;
    }
    else  // if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::Dense )
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
 * @see hasnan( access_t accessType, const matrix_t& A ).
 */
template <class matrix_t>
bool hasnan(band_t accessType, const matrix_t& A)
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
template <class vector_t>
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
 */
template <typename T>
T abs(const T& x);

inline float abs(float x) { return std::fabs(x); }
inline double abs(double x) { return std::fabs(x); }
inline long double abs(long double x) { return std::fabs(x); }

template <typename T>
inline T abs(const std::complex<T>& x)
{
    // If the default value of ErrorCheck::nan is true then check for NaNs
    return (ErrorCheck().nan)
               ? (isnan(x) ? std::numeric_limits<T>::quiet_NaN() : std::abs(x))
               : std::abs(x);
}

// -----------------------------------------------------------------------------
/// 1-norm absolute value, |Re(x)| + |Im(x)|
template <typename real_t>
inline real_t abs1(const real_t& x)
{
    return abs(x);
}

template <typename real_t>
inline real_t abs1(const std::complex<real_t>& x)
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
        is_scalar<C1> || is_scalar<C2> || (layout<C1> == Layout::Strided) ||
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
    struct AllowOptBLASImpl {
        static constexpr bool value =
            false;  ///< True if the list of types
                    ///< allows optimized BLAS library.
    };
}  // namespace internal

/// Alias for @c internal::AllowOptBLASImpl<,int>::value.
/// @ingroup abstract_matrix
template <class... Ts>
constexpr bool allow_optblas = internal::AllowOptBLASImpl<Ts..., int>::value;

namespace internal {

    /// True if C is a row- or column-major matrix and the entry type can be
    /// used with optimized BLAS implementations.
    template <class C>
    struct AllowOptBLASImpl<C, enable_if_t<is_matrix<C>, int>> {
        static constexpr bool value =
            allow_optblas<type_t<C>> &&
            (layout<C> == Layout::ColMajor || layout<C> == Layout::RowMajor);
    };

    /// True if C is a strided vector and the entry type can be used with
    /// optimized BLAS implementations.
    template <class C>
    struct AllowOptBLASImpl<C, enable_if_t<is_vector<C>, int>> {
        static constexpr bool value =
            allow_optblas<type_t<C>> && (layout<C> == Layout::Strided);
    };

    /** A pair of types <C,T> allows optimized BLAS if T allows optimized BLAS
     * and one of the followings happens:
     * 1. C is a matrix or vector that allows optimized BLAS and the entry type
     * is the same as T.
     * 2. C is not a matrix or vector, but is convertible to T.
     */
    template <class C, class T>
    struct AllowOptBLASImpl<pair<C, T>, int> {
        static constexpr bool value =
            allow_optblas<T> &&
            ((is_matrix<C> || is_vector<C>)
                 ? (allow_optblas<C> &&
                    is_same_v<type_t<C>, typename std::decay<T>::type>)
                 : std::is_convertible<C, T>::value);
    };

    template <class C1, class T1, class C2, class T2, class... Ps>
    struct AllowOptBLASImpl<pair<C1, T1>, pair<C2, T2>, Ps...> {
        static constexpr bool value =
            AllowOptBLASImpl<pair<C1, T1>, int>::value &&
            AllowOptBLASImpl<pair<C2, T2>, Ps...>::value &&
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
        struct AllowOptBLASImpl<T, int> {       \
            static constexpr bool value = true; \
        };                                      \
        template <>                             \
        struct type_trait<T> {                  \
            using type = T;                     \
        };                                      \
    }

/// Optimized types
#ifdef USE_LAPACKPP_WRAPPERS
TLAPACK_OPT_TYPE(float)
TLAPACK_OPT_TYPE(double)
TLAPACK_OPT_TYPE(std::complex<float>)
TLAPACK_OPT_TYPE(std::complex<double>)
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
inline Workspace alloc_workspace(vectorOfBytes& v, std::size_t lwork)
{
    v = vectorOfBytes(lwork);  // Allocates space in memory
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
inline Workspace alloc_workspace(vectorOfBytes& v,
                                 const workinfo_t& workinfo,
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

/// Alias for @c deduce_work<>::type
template <class work_type, class work_default>
using deduce_work_t = typename deduce_work<work_type, work_default>::type;

/**
 * @brief Options structure with a Workspace attribute
 *
 * @tparam work_t Give specialized data type to the workspaces.
 *      Behavior defined by each implementation using this option.
 */
template <class... work_t>
struct workspace_opts_t {
    Workspace work;  ///< Workspace object

    // Constructors:

    inline constexpr workspace_opts_t(Workspace&& w = {}) : work(w) {}

    inline constexpr workspace_opts_t(const Workspace& w) : work(w) {}

    template <class matrix_t>
    inline constexpr workspace_opts_t(const matrix_t& A)
        : work(legacy_matrix(A))
    {}
};

}  // namespace tlapack

#endif  // TLAPACK_UTILS_HH
