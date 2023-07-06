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

#include <vector>

#include "tlapack/base/StrongZero.hpp"
#include "tlapack/base/scalar_type_traits.hpp"

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

namespace internal {
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
}  // namespace internal

// constant expressions for upper/lower access

/// General access
constexpr internal::generalAccess_t dense = {};
/// Upper Hessenberg access
constexpr internal::upperHessenberg_t upperHessenberg = {};
/// Lower Hessenberg access
constexpr internal::lowerHessenberg_t lowerHessenberg = {};
/// Upper Triangle access
constexpr internal::upperTriangle_t upperTriangle = {};
/// Lower Triangle access
constexpr internal::lowerTriangle_t lowerTriangle = {};
/// Strict Upper Triangle access
constexpr internal::strictUpper_t strictUpper = {};
/// Strict Lower Triangle access
constexpr internal::strictLower_t strictLower = {};

// -----------------------------------------------------------------------------
// Information about the main diagonal

enum class Diag : char {
    NonUnit = 'N',  ///< The main diagonal is not assumed to consist of 1's.
    Unit = 'U'      ///< The main diagonal is assumed to consist of 1's.
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Diag, NonUnit, Unit)

namespace internal {
    struct nonUnit_diagonal_t {
        constexpr operator Diag() const { return Diag::NonUnit; }
    };
    struct unit_diagonal_t {
        constexpr operator Diag() const { return Diag::Unit; }
    };
}  // namespace internal

// constant expressions about the main diagonal

/// The main diagonal is not assumed to consist of 1's.
constexpr internal::nonUnit_diagonal_t nonUnit_diagonal = {};
/// The main diagonal is assumed to consist of 1's.
constexpr internal::unit_diagonal_t unit_diagonal = {};

// -----------------------------------------------------------------------------
// Operations over data

enum class Op : char {
    NoTrans = 'N',    ///< no transpose
    Trans = 'T',      ///< transpose
    ConjTrans = 'C',  ///< conjugate transpose
    Conj = 3          ///< non-transpose conjugate
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_4_VALUES(Op, NoTrans, Trans, ConjTrans, Conj)

namespace internal {
    struct noTranspose_t {
        constexpr operator Op() const { return Op::NoTrans; }
    };
    struct transpose_t {
        constexpr operator Op() const { return Op::Trans; }
    };
    struct conjTranspose_t {
        constexpr operator Op() const { return Op::ConjTrans; }
    };
}  // namespace internal

// Constant expressions for operations over data

/// no transpose
constexpr internal::noTranspose_t noTranspose = {};
/// transpose
constexpr internal::transpose_t Transpose = {};
/// conjugate transpose
constexpr internal::conjTranspose_t conjTranspose = {};

// -----------------------------------------------------------------------------
// Sides

enum class Side : char {
    Left = 'L',  ///< left side
    Right = 'R'  ///< right side
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Side, Left, Right)

namespace internal {
    struct left_side_t {
        constexpr operator Side() const { return Side::Left; }
    };
    struct right_side_t {
        constexpr operator Side() const { return Side::Right; }
    };
}  // namespace internal

// Constant expressions for sides

/// left side
constexpr internal::left_side_t left_side{};
/// right side
constexpr internal::right_side_t right_side{};

// -----------------------------------------------------------------------------
// Norm types

enum class Norm : char {
    One = '1',  ///< one norm
    Two = '2',  ///< two norm
    Inf = 'I',  ///< infinity norm of matrices
    Fro = 'F',  ///< Frobenius norm of matrices
    Max = 'M',  ///< max norm
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_5_VALUES(Norm, One, Two, Inf, Fro, Max)

namespace internal {
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
}  // namespace internal

// Constant expressions for norm types

/// max norm
constexpr internal::max_norm_t max_norm = {};
/// one norm
constexpr internal::one_norm_t one_norm = {};
/// two norm
constexpr internal::two_norm_t two_norm = {};
/// infinity norm of matrices
constexpr internal::inf_norm_t inf_norm = {};
/// Frobenius norm of matrices
constexpr internal::frob_norm_t frob_norm = {};

// -----------------------------------------------------------------------------
// Directions

enum class Direction : char {
    Forward = 'F',   ///< Forward direction
    Backward = 'B',  ///< Backward direction
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Direction, Forward, Backward)

namespace internal {
    struct forward_t {
        constexpr operator Direction() const { return Direction::Forward; }
    };
    struct backward_t {
        constexpr operator Direction() const { return Direction::Backward; }
    };
}  // namespace internal

// Constant expressions for directions

/// Forward direction
constexpr internal::forward_t forward{};
/// Backward direction
constexpr internal::backward_t backward{};

// -----------------------------------------------------------------------------
// Storage types

enum class StoreV : char {
    Columnwise = 'C',  ///< Columnwise storage
    Rowwise = 'R',     ///< Rowwise storage
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(StoreV, Columnwise, Rowwise)

namespace internal {
    struct columnwise_storage_t {
        constexpr operator StoreV() const { return StoreV::Columnwise; }
    };
    struct rowwise_storage_t {
        constexpr operator StoreV() const { return StoreV::Rowwise; }
    };
}  // namespace internal

// Constant expressions for storage types

/// Columnwise storage
constexpr internal::columnwise_storage_t columnwise_storage{};
/// Rowwise storage
constexpr internal::rowwise_storage_t rowwise_storage{};

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
struct BandAccess {
    std::size_t lower_bandwidth;  ///< Number of subdiagonals.
    std::size_t upper_bandwidth;  ///< Number of superdiagonals.

    /**
     * @brief Construct a new Band Access object
     *
     * @param kl Number of subdiagonals.
     * @param ku Number of superdiagonals.
     */
    constexpr BandAccess(std::size_t kl, std::size_t ku)
        : lower_bandwidth(kl), upper_bandwidth(ku)
    {}
};

// -----------------------------------------------------------------------------
// Workspace

/// Byte type
using byte = unsigned char;
/// Vector of bytes. May use a specialized allocator in future
using VectorOfBytes = std::vector<byte, std::allocator<byte>>;

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
    struct Matrix {
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
    struct Vector {
        idx_t n;
        T* ptr;
        idx_t inc;
    };
}  // namespace legacy
}  // namespace tlapack

#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES
#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_4_VALUES
#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_5_VALUES
#undef TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_7_VALUES

#endif  // TLAPACK_TYPES_HH
