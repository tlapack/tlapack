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

// Helpers:

#define TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(EnumClass, A, B)       \
    inline std::ostream& operator<<(std::ostream& out, const EnumClass v) \
    {                                                                     \
        if (v == EnumClass::A) return out << #A;                          \
        if (v == EnumClass::B) return out << #B;                          \
        return out << "<Invalid>";                                        \
    }

#define TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_3_VALUES(EnumClass, A, B, C)    \
    inline std::ostream& operator<<(std::ostream& out, const EnumClass v) \
    {                                                                     \
        if (v == EnumClass::A) return out << #A;                          \
        if (v == EnumClass::B) return out << #B;                          \
        if (v == EnumClass::C) return out << #C;                          \
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

// Types:

namespace tlapack {

namespace internal {

    /// Auxiliary data type to vector increments.
    struct StrongOne {
        inline constexpr operator int() const { return 1; }
        inline constexpr StrongOne(int i = 1) { assert(i == 1); }
    };

    /**
     * @brief Auxiliary data type
     *
     * Suppose x is of type T. Then:
     *
     * 1. T(StrongZero()) is equivalent to T(0).
     * 2. x *= StrongZero() is equivalent to x = T(0).
     * 3. x += StrongZero() does not modify x.
     *
     * This class satisfies:
     *
     *      x * StrongZero() = StrongZero()
     *      StrongZero() * x = StrongZero()
     *      x + StrongZero() = x
     *      StrongZero() + x = x
     *
     */
    struct StrongZero {
        template <typename T>
        explicit constexpr operator T() const
        {
            return T(0);
        }

        template <typename T>
        friend constexpr T& operator*=(T& lhs, const StrongZero&)
        {
            lhs = T(0);
            return lhs;
        }

        template <typename T>
        friend constexpr const StrongZero operator*(const StrongZero&, const T&)
        {
            return StrongZero();
        }

        template <typename T>
        friend constexpr const StrongZero operator*(const T&, const StrongZero&)
        {
            return StrongZero();
        }

        template <typename T>
        friend constexpr const T& operator+=(const T& lhs, const StrongZero&)
        {
            return lhs;
        }

        template <typename T>
        friend constexpr const T operator+(const StrongZero&, const T& rhs)
        {
            return rhs;
        }

        template <typename T>
        friend constexpr const T operator+(const T& lhs, const StrongZero&)
        {
            return lhs;
        }
    };
}  // namespace internal

// -----------------------------------------------------------------------------
// Layouts

enum class Layout : char {
    Unspecified = 0,
    ColMajor = 'C',
    RowMajor = 'R',
    BandStorage = 'B'
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_4_VALUES(
    Layout, Unspecified, ColMajor, RowMajor, BandStorage);

// -----------------------------------------------------------------------------
// Upper or Lower access

enum class Uplo : char { Upper = 'U', Lower = 'L', General = 'G' };
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_3_VALUES(Uplo, Upper, Lower, General);

// -----------------------------------------------------------------------------
// Information about the main diagonal

enum class Diag : char { NonUnit = 'N', Unit = 'U' };
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Diag, NonUnit, Unit);

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
    Conj = 0  ///< non-transpose conjugate
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_4_VALUES(Op, NoTrans, Trans, ConjTrans, Conj);

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
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Side, Left, Right);

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
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_5_VALUES(Norm, One, Two, Inf, Fro, Max);

struct max_norm_t {
    constexpr operator Norm() const { return Norm::Max; }
};
struct one_norm_t {
    constexpr operator Norm() const { return Norm::One; }
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
constexpr inf_norm_t inf_norm = {};
constexpr frob_norm_t frob_norm = {};

// -----------------------------------------------------------------------------
// Directions

enum class Direction : char {
    Forward = 'F',
    Backward = 'B',
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Direction, Forward, Backward);

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
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(StoreV, Columnwise, Rowwise);

struct columnwise_storage_t {
    constexpr operator StoreV() const { return StoreV::Columnwise; }
};
struct rowwise_storage_t {
    constexpr operator StoreV() const { return StoreV::Rowwise; }
};

// Constants
constexpr columnwise_storage_t columnwise_storage{};
constexpr rowwise_storage_t rowwise_storage{};
}  // namespace tlapack

#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES
#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_3_VALUES
#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_4_VALUES
#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_5_VALUES

namespace tlapack {

// -----------------------------------------------------------------------------
// Based on C++14 std::common_type implementation from
// http://www.cplusplus.com/reference/type_traits/std::common_type/
// Adds promotion of complex types based on the common type of the associated
// real types. This fixes various cases:
//
// std::std::common_type_t< double, complex<float> > is complex<float>  (wrong)
//        scalar_type< double, complex<float> > is complex<double> (right)
//
// std::std::common_type_t< int, complex<long> > is not defined (compile error)
//        scalar_type< int, complex<long> > is complex<long> (right)

// for zero types
template <typename... Types>
struct scalar_type_traits;

/// define scalar_type<> type alias
template <typename... Types>
using scalar_type = typename scalar_type_traits<Types...>::type;

// for one type
template <typename T>
struct scalar_type_traits<T> {
    using type = typename std::decay<T>::type;
};

// for two types
// relies on type of ?: operator being the common type of its two arguments
template <typename T1, typename T2>
struct scalar_type_traits<T1, T2> {
    using type = typename std::decay<decltype(true ? std::declval<T1>()
                                                   : std::declval<T2>())>::type;
};

// for either or both complex,
// find common type of associated real types, then add complex
template <typename T1, typename T2>
struct scalar_type_traits<std::complex<T1>, T2> {
    using type = std::complex<typename std::common_type<T1, T2>::type>;
};

template <typename T1, typename T2>
struct scalar_type_traits<T1, std::complex<T2> > {
    using type = std::complex<typename std::common_type<T1, T2>::type>;
};

template <typename T1, typename T2>
struct scalar_type_traits<std::complex<T1>, std::complex<T2> > {
    using type = std::complex<typename std::common_type<T1, T2>::type>;
};

// for three or more types
template <typename T1, typename T2, typename... Types>
struct scalar_type_traits<T1, T2, Types...> {
    using type = scalar_type<scalar_type<T1, T2>, Types...>;
};

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

// for zero types
template <typename... Types>
struct real_type_traits;

/// define real_type<> type alias
template <typename... Types>
using real_type = typename real_type_traits<Types...>::type;

// for one type
template <typename T>
struct real_type_traits<T> {
    using type = T;
};

// for one complex type, strip complex
template <typename T>
struct real_type_traits<std::complex<T> > {
    using type = T;
};

// for two or more types
template <typename T1, typename... Types>
struct real_type_traits<T1, Types...> {
    using type = scalar_type<real_type<T1>, real_type<Types...> >;
};

// for zero types
template <typename... Types>
struct complex_type_traits;

/// define complex_type<> type alias
template <typename... Types>
using complex_type = typename complex_type_traits<Types...>::type;

// for one type
template <typename T>
struct complex_type_traits<T> {
    using type = std::complex<T>;
};

// for one complex type, strip complex
template <typename T>
struct complex_type_traits<std::complex<T> > {
    using type = std::complex<T>;
};

// for two or more types
template <typename T1, typename... Types>
struct complex_type_traits<T1, Types...> {
    using type = scalar_type<complex_type<T1>, complex_type<Types...> >;
};

}  // namespace tlapack

namespace tlapack {

namespace internal {

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

}  // namespace tlapack

namespace tlapack {

// Workspace

/// Byte type
using byte = unsigned char;
/// Byte allocator
using byteAlloc = std::allocator<byte>;
/// Vector of bytes. May use a specialized allocator in future
using vectorOfBytes = std::vector<byte, byteAlloc>;

}  // namespace tlapack

#endif  // TLAPACK_TYPES_HH
