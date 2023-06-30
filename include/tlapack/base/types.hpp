/// @file types.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TYPES_HH
#define TLAPACK_TYPES_HH

#include <cassert>
#include <complex>
#include <type_traits>
#include <vector>

#include "tlapack/base/StrongZero.hpp"

// Helpers:

#define TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(EnumClass, A, B)       \
    inline std::ostream& operator<<(std::ostream& out, const EnumClass v) \
    {                                                                     \
        if (v == EnumClass::A) return out << #A;                          \
        if (v == EnumClass::B) return out << #B;                          \
        return out << "<Invalid>";                                        \
    }

#define TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_4_VALUES(EnumClass, A, B, C, D) \
    inline std::ostream& operator<<(std::ostream& out, const EnumClass v) \
    {                                                                     \
        if (v == EnumClass::A) return out << #A;                          \
        if (v == EnumClass::B) return out << #B;                          \
        if (v == EnumClass::C) return out << #C;                          \
        if (v == EnumClass::D) return out << #D;                          \
        return out << "<Invalid>";                                        \
    }

#define TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_5_VALUES(EnumClass, A, B, C, D, E) \
    inline std::ostream& operator<<(std::ostream& out, const EnumClass v)    \
    {                                                                        \
        if (v == EnumClass::A) return out << #A;                             \
        if (v == EnumClass::B) return out << #B;                             \
        if (v == EnumClass::C) return out << #C;                             \
        if (v == EnumClass::D) return out << #D;                             \
        if (v == EnumClass::E) return out << #E;                             \
        return out << "<Invalid>";                                           \
    }

#define TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_7_VALUES(EnumClass, A, B, C, D, E, \
                                                   F, G)                     \
    inline std::ostream& operator<<(std::ostream& out, const EnumClass v)    \
    {                                                                        \
        if (v == EnumClass::A) return out << #A;                             \
        if (v == EnumClass::B) return out << #B;                             \
        if (v == EnumClass::C) return out << #C;                             \
        if (v == EnumClass::D) return out << #D;                             \
        if (v == EnumClass::E) return out << #E;                             \
        if (v == EnumClass::F) return out << #F;                             \
        if (v == EnumClass::G) return out << #G;                             \
        return out << "<Invalid>";                                           \
    }

// Types:

namespace tlapack {

// -----------------------------------------------------------------------------
// Layouts

enum class Layout : char {
    Strided = 'S',   ///< Strided layout. Vectors whose i-th element is at
                     ///< ptr + i*inc, inc is an integer.
    ColMajor = 'C',  ///< Column-major layout. Matrices whose (i,j)-th element
                     ///< is at ptr + i + j*ldim.
    RowMajor = 'R',  ///< Row-major layout. Matrices whose (i,j)-th element
                     ///< is at ptr + i*ldim + j.
    Unspecified = 0  ///< Used on all other data structures.
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_4_VALUES(
    Layout, Unspecified, ColMajor, RowMajor, Strided)

// -----------------------------------------------------------------------------
// Upper or Lower access

enum class Uplo : char {
    General = 'G',          ///< 0 <= i <= m,   0 <= j <= n.
    Upper = 'U',            ///< 0 <= i <= j,   0 <= j <= n.
    Lower = 'L',            ///< 0 <= i <= m,   0 <= j <= i.
    UpperHessenberg = 'H',  ///< 0 <= i <= j+1, 0 <= j <= n.
    LowerHessenberg = 4,    ///< 0 <= i <= m,   0 <= j <= i+1.
    StrictUpper = 'S',      ///< 0 <= i <= j-1, 0 <= j <= n.
    StrictLower = 6,        ///< 0 <= i <= m,   0 <= j <= i-1.
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_7_VALUES(Uplo,
                                           Upper,
                                           Lower,
                                           General,
                                           UpperHessenberg,
                                           LowerHessenberg,
                                           StrictUpper,
                                           StrictLower)

/**
 * @brief General access.
 *
 * Pairs (i,j) such that 0 <= i <= m,   0 <= j <= n     in a m-by-n matrix.
 *
 *      x x x x x
 *      x x x x x
 *      x x x x x
 *      x x x x x
 */
struct generalAccess_t {
    constexpr operator Uplo() const { return Uplo::General; }
};

/**
 * @brief Upper Triangle access
 *
 * Pairs (i,j) such that 0 <= i <= j,   0 <= j <= n     in a m-by-n matrix.
 *
 *      x x x x x
 *      0 x x x x
 *      0 0 x x x
 *      0 0 0 x x
 */
struct upperTriangle_t {
    constexpr operator Uplo() const { return Uplo::Upper; }
};

/**
 * @brief Lower Triangle access
 *
 * Pairs (i,j) such that 0 <= i <= m,   0 <= j <= i     in a m-by-n matrix.
 *
 *      x 0 0 0 0
 *      x x 0 0 0
 *      x x x 0 0
 *      x x x x 0
 */
struct lowerTriangle_t {
    constexpr operator Uplo() const { return Uplo::Lower; }
};

/**
 * @brief Upper Hessenberg access
 *
 * Pairs (i,j) such that 0 <= i <= j+1, 0 <= j <= n     in a m-by-n matrix.
 *
 *      x x x x x
 *      x x x x x
 *      0 x x x x
 *      0 0 x x x
 */
struct upperHessenberg_t {
    constexpr operator Uplo() const { return Uplo::UpperHessenberg; }
};

/**
 * @brief Lower Hessenberg access
 *
 * Pairs (i,j) such that 0 <= i <= m,   0 <= j <= i+1   in a m-by-n matrix.
 *
 *      x x 0 0 0
 *      x x x 0 0
 *      x x x x 0
 *      x x x x x
 */
struct lowerHessenberg_t {
    constexpr operator Uplo() const { return Uplo::LowerHessenberg; }
};

/**
 * @brief Strict Upper Triangle access
 *
 * Pairs (i,j) such that 0 <= i <= j-1, 0 <= j <= n     in a m-by-n matrix.
 *
 *      0 x x x x
 *      0 0 x x x
 *      0 0 0 x x
 *      0 0 0 0 x
 */
struct strictUpper_t {
    constexpr operator Uplo() const { return Uplo::StrictUpper; }
};

/**
 * @brief Strict Lower Triangle access
 *
 * Pairs (i,j) such that 0 <= i <= m,   0 <= j <= i-1   in a m-by-n matrix.
 *
 *      0 0 0 0 0
 *      x 0 0 0 0
 *      x x 0 0 0
 *      x x x 0 0
 */
struct strictLower_t {
    constexpr operator Uplo() const { return Uplo::StrictLower; }
};

// constant expressions
constexpr generalAccess_t dense = {};
constexpr upperHessenberg_t upperHessenberg = {};
constexpr lowerHessenberg_t lowerHessenberg = {};
constexpr upperTriangle_t upperTriangle = {};
constexpr lowerTriangle_t lowerTriangle = {};
constexpr strictUpper_t strictUpper = {};
constexpr strictLower_t strictLower = {};

// -----------------------------------------------------------------------------
// Information about the main diagonal

enum class Diag : char { NonUnit = 'N', Unit = 'U' };
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Diag, NonUnit, Unit)

struct nonUnit_diagonal_t {
    constexpr operator Diag() const { return Diag::NonUnit; }
};
struct unit_diagonal_t {
    constexpr operator Diag() const { return Diag::Unit; }
};

// constants
constexpr nonUnit_diagonal_t nonUnit_diagonal = {};
constexpr unit_diagonal_t unit_diagonal = {};

// -----------------------------------------------------------------------------
// Operations over data

enum class Op : char {
    NoTrans = 'N',
    Trans = 'T',
    ConjTrans = 'C',
    Conj = 3  ///< non-transpose conjugate
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_4_VALUES(Op, NoTrans, Trans, ConjTrans, Conj)

struct noTranspose_t {
    constexpr operator Op() const { return Op::NoTrans; }
};
struct transpose_t {
    constexpr operator Op() const { return Op::Trans; }
};
struct conjTranspose_t {
    constexpr operator Op() const { return Op::ConjTrans; }
};

// Constants
constexpr noTranspose_t noTranspose = {};
constexpr transpose_t Transpose = {};
constexpr conjTranspose_t conjTranspose = {};

// -----------------------------------------------------------------------------
// Sides

enum class Side : char { Left = 'L', Right = 'R' };
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Side, Left, Right)

struct left_side_t {
    constexpr operator Side() const { return Side::Left; }
};
struct right_side_t {
    constexpr operator Side() const { return Side::Right; }
};

// Constants
constexpr left_side_t left_side{};
constexpr right_side_t right_side{};

// -----------------------------------------------------------------------------
// Norm types

enum class Norm : char {
    One = '1',  // or 'O'
    Two = '2',
    Inf = 'I',
    Fro = 'F',  // or 'E'
    Max = 'M',
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_5_VALUES(Norm, One, Two, Inf, Fro, Max)

struct max_norm_t {
    constexpr operator Norm() const { return Norm::Max; }
};
struct one_norm_t {
    constexpr operator Norm() const { return Norm::One; }
};
struct two_norm_t {
    constexpr operator Norm() const { return Norm::Two; }
};
struct inf_norm_t {
    constexpr operator Norm() const { return Norm::Inf; }
};
struct frob_norm_t {
    constexpr operator Norm() const { return Norm::Fro; }
};

// Constants
constexpr max_norm_t max_norm = {};
constexpr one_norm_t one_norm = {};
constexpr two_norm_t two_norm = {};
constexpr inf_norm_t inf_norm = {};
constexpr frob_norm_t frob_norm = {};

// -----------------------------------------------------------------------------
// Directions

enum class Direction : char {
    Forward = 'F',
    Backward = 'B',
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Direction, Forward, Backward)

struct forward_t {
    constexpr operator Direction() const { return Direction::Forward; }
};
struct backward_t {
    constexpr operator Direction() const { return Direction::Backward; }
};

// Constants
constexpr forward_t forward{};
constexpr backward_t backward{};

// -----------------------------------------------------------------------------
// Storage types

enum class StoreV : char {
    Columnwise = 'C',
    Rowwise = 'R',
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(StoreV, Columnwise, Rowwise)

struct columnwise_storage_t {
    constexpr operator StoreV() const { return StoreV::Columnwise; }
};
struct rowwise_storage_t {
    constexpr operator StoreV() const { return StoreV::Rowwise; }
};

// Constants
constexpr columnwise_storage_t columnwise_storage{};
constexpr rowwise_storage_t rowwise_storage{};

// -----------------------------------------------------------------------------
// Band access

/**
 * @brief Band access
 *
 * Pairs (i,j) such that max(0,j-ku) <= i <= min(m,j+kl) in a m-by-n matrix,
 * where kl is the lower_bandwidth and ku is the upper_bandwidth.
 *
 *      x x x 0 0
 *      x x x x 0
 *      0 x x x x
 *      0 0 x x x
 */
struct band_t {
    std::size_t lower_bandwidth;  ///< Number of subdiagonals.
    std::size_t upper_bandwidth;  ///< Number of superdiagonals.

    constexpr band_t(std::size_t kl, std::size_t ku)
        : lower_bandwidth(kl), upper_bandwidth(ku)
    {}
};

}  // namespace tlapack

#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES
#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_4_VALUES
#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_5_VALUES
#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_7_VALUES

namespace tlapack {

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

namespace internal {
    template <typename... Types>
    struct real_type_traits;

    template <typename... Types>
    struct complex_type_traits;

    template <typename... Types>
    struct scalar_type_traits;
}  // namespace internal

/// define real_type<> type alias
template <typename... Types>
using real_type = typename internal::real_type_traits<Types..., int>::type;

/// define complex_type<> type alias
template <typename... Types>
using complex_type =
    typename internal::complex_type_traits<Types..., int>::type;

/// define scalar_type<> type alias
template <typename... Types>
using scalar_type = typename internal::scalar_type_traits<Types..., int>::type;

/// True if T is real_type<T>
template <typename T>
struct is_real {
    static constexpr bool value =
        std::is_same_v<real_type<T>, typename std::decay<T>::type>;
};

/// True if T is complex_type<T>
template <typename T>
struct is_complex {
    static constexpr bool value =
        std::is_same_v<complex_type<T>, typename std::decay<T>::type>;
};

namespace internal {

    // for one std arithmetic type
    template <typename T>
    struct real_type_traits<
        T,
        std::enable_if_t<std::is_arithmetic_v<T> && !std::is_const_v<T>, int>> {
        using type = typename std::decay<T>::type;
    };

    template <typename T>
    struct real_type_traits<const T, int> {
        using type = real_type<T>;
    };

    // for one complex type, strip complex
    template <typename T>
    struct real_type_traits<std::complex<T>, int> {
        using type = real_type<T>;
    };

    // pointers and references don't have a real type
    template <typename T>
    struct real_type_traits<
        T,
        std::enable_if_t<std::is_pointer_v<T> || std::is_reference_v<T>, int>> {
        using type = void;
    };

    // for two or more types
    template <typename T1, typename T2, typename... Types>
    struct real_type_traits<T1, T2, Types...> {
        using type =
            std::common_type_t<typename real_type_traits<T1, int>::type,
                               typename real_type_traits<T2, Types...>::type>;
    };

    // for one std arithmetic type
    template <typename T>
    struct complex_type_traits<
        T,
        std::enable_if_t<std::is_arithmetic_v<T> && !std::is_const_v<T>, int>> {
        using type = std::complex<real_type<T>>;
    };

    template <typename T>
    struct complex_type_traits<const T, int> {
        using type = complex_type<T>;
    };

    // for one complex type, strip complex
    template <typename T>
    struct complex_type_traits<std::complex<T>, int> {
        using type = std::complex<real_type<T>>;
    };

    // pointers and references don't have a complex type
    template <typename T>
    struct complex_type_traits<
        T,
        std::enable_if_t<std::is_pointer_v<T> || std::is_reference_v<T>, int>> {
        using type = void;
    };

    // for two or more types
    template <typename T1, typename T2, typename... Types>
    struct complex_type_traits<T1, T2, Types...> {
        using type =
            complex_type<typename real_type_traits<T1, T2, Types...>::type>;
    };

    // for one type
    template <typename T>
    struct scalar_type_traits<T, int> : scalar_type_traits<T, T, int> {};

    // for two types, one is complex
    template <typename T1, typename T2>
    struct scalar_type_traits<
        T1,
        T2,
        std::enable_if_t<is_complex<T1>::value || is_complex<T2>::value, int>> {
        using type = complex_type<T1, T2>;
    };

    // for two types, neither is complex
    template <typename T1, typename T2>
    struct scalar_type_traits<
        T1,
        T2,
        std::enable_if_t<is_real<T1>::value && is_real<T2>::value, int>> {
        using type = real_type<T1, T2>;
    };

    // for three or more types
    template <typename T1, typename T2, typename... Types>
    struct scalar_type_traits<T1, T2, Types...> {
        using type =
            typename scalar_type_traits<scalar_type<T1, T2>, Types...>::type;
    };

    /**
     * @brief Data type trait.
     *
     * The data type is defined on @c type_trait<array_t>::type.
     *
     * @tparam T A non-array class.
     */
    template <class T, typename = int>
    struct type_trait {
        using type = void;
    };

    /**
     * @brief Size type trait.
     *
     * The size type is defined on @c sizet_trait<array_t>::type.
     *
     * @tparam T A non-array class.
     */
    template <class T, typename = int>
    struct sizet_trait {
        using type = std::size_t;
    };

}  // namespace internal

/// Alias for @c type_trait<>::type.
template <class array_t>
using type_t = typename internal::type_trait<array_t>::type;

/// Alias for @c sizet_trait<>::type.
template <class array_t>
using size_type = typename internal::sizet_trait<array_t>::type;

// Workspace

/// Byte type
using byte = unsigned char;
/// Byte allocator
using byteAlloc = std::allocator<byte>;
/// Vector of bytes. May use a specialized allocator in future
using vectorOfBytes = std::vector<byte, byteAlloc>;

// -----------------------------------------------------------------------------
// Legacy matrix and vector structures

namespace legacy {

    /**
     * @brief Describes a row- or column-major matrix
     *
     * @tparam T Type of each entry.
     * @tparam idx_t Integer type of the size attributes.
     */
    template <class T, class idx_t>
    struct matrix {
        Layout layout;
        idx_t m;
        idx_t n;
        T* ptr;
        idx_t ldim;
    };

    /**
     * @brief Describes a strided vector.
     *
     * @tparam T Type of each entry.
     * @tparam idx_t Integer type of the size attributes.
     */
    template <class T, class idx_t>
    struct vector {
        idx_t n;
        T* ptr;
        idx_t inc;
    };

}  // namespace legacy

}  // namespace tlapack

#endif  // TLAPACK_TYPES_HH
