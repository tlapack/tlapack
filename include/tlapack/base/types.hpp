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
    struct GeneralAccess {
        constexpr operator Uplo() const noexcept { return Uplo::General; }
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
    struct UpperTriangle {
        constexpr operator Uplo() const noexcept { return Uplo::Upper; }
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
    struct LowerTriangle {
        constexpr operator Uplo() const noexcept { return Uplo::Lower; }
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
    struct UpperHessenberg {
        constexpr operator Uplo() const noexcept
        {
            return Uplo::UpperHessenberg;
        }
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
    struct LowerHessenberg {
        constexpr operator Uplo() const noexcept
        {
            return Uplo::LowerHessenberg;
        }
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
    struct StrictUpper {
        constexpr operator Uplo() const noexcept { return Uplo::StrictUpper; }
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
    struct StrictLower {
        constexpr operator Uplo() const noexcept { return Uplo::StrictLower; }
    };
}  // namespace internal

// constant expressions for upper/lower access

/// General access
constexpr internal::GeneralAccess GENERAL = {};
/// Upper Hessenberg access
constexpr internal::UpperHessenberg UPPER_HESSENBERG = {};
/// Lower Hessenberg access
constexpr internal::LowerHessenberg LOWER_HESSENBERG = {};
/// Upper Triangle access
constexpr internal::UpperTriangle UPPER_TRIANGLE = {};
/// Lower Triangle access
constexpr internal::LowerTriangle LOWER_TRIANGLE = {};
/// Strict Upper Triangle access
constexpr internal::StrictUpper STRICT_UPPER = {};
/// Strict Lower Triangle access
constexpr internal::StrictLower STRICT_LOWER = {};

// -----------------------------------------------------------------------------
// Information about the main diagonal

enum class Diag : char {
    NonUnit = 'N',  ///< The main diagonal is not assumed to consist of 1's.
    Unit = 'U'      ///< The main diagonal is assumed to consist of 1's.
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Diag, NonUnit, Unit)

namespace internal {
    struct NonUnitDiagonal {
        constexpr operator Diag() const noexcept { return Diag::NonUnit; }
    };
    struct UnitDiagonal {
        constexpr operator Diag() const noexcept { return Diag::Unit; }
    };
}  // namespace internal

// constant expressions about the main diagonal

/// The main diagonal is not assumed to consist of 1's.
constexpr internal::NonUnitDiagonal NON_UNIT_DIAG = {};
/// The main diagonal is assumed to consist of 1's.
constexpr internal::UnitDiagonal UNIT_DIAG = {};

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
    struct NoTranspose {
        constexpr operator Op() const noexcept { return Op::NoTrans; }
    };
    struct Transpose {
        constexpr operator Op() const noexcept { return Op::Trans; }
    };
    struct ConjTranspose {
        constexpr operator Op() const noexcept { return Op::ConjTrans; }
    };
    struct Conjugate {
        constexpr operator Op() const noexcept { return Op::Conj; }
    };
}  // namespace internal

// Constant expressions for operations over data

/// no transpose
constexpr internal::NoTranspose NO_TRANS = {};
/// transpose
constexpr internal::Transpose TRANSPOSE = {};
/// conjugate transpose
constexpr internal::ConjTranspose CONJ_TRANS = {};
/// non-transpose conjugate
constexpr internal::Conjugate CONJUGATE = {};

// -----------------------------------------------------------------------------
// Sides

enum class Side : char {
    Left = 'L',  ///< left side
    Right = 'R'  ///< right side
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Side, Left, Right)

namespace internal {
    struct LeftSide {
        constexpr operator Side() const noexcept { return Side::Left; }
    };
    struct RightSide {
        constexpr operator Side() const noexcept { return Side::Right; }
    };
}  // namespace internal

// Constant expressions for sides

/// left side
constexpr internal::LeftSide LEFT_SIDE{};
/// right side
constexpr internal::RightSide RIGHT_SIDE{};

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
    struct MaxNorm {
        constexpr operator Norm() const noexcept { return Norm::Max; }
    };
    struct OneNorm {
        constexpr operator Norm() const noexcept { return Norm::One; }
    };
    struct TwoNorm {
        constexpr operator Norm() const noexcept { return Norm::Two; }
    };
    struct InfNorm {
        constexpr operator Norm() const noexcept { return Norm::Inf; }
    };
    struct FrobNorm {
        constexpr operator Norm() const noexcept { return Norm::Fro; }
    };
}  // namespace internal

// Constant expressions for norm types

/// max norm
constexpr internal::MaxNorm MAX_NORM = {};
/// one norm
constexpr internal::OneNorm ONE_NORM = {};
/// two norm
constexpr internal::TwoNorm TWO_NORM = {};
/// infinity norm of matrices
constexpr internal::InfNorm INF_NORM = {};
/// Frobenius norm of matrices
constexpr internal::FrobNorm FROB_NORM = {};

// -----------------------------------------------------------------------------
// Directions

enum class Direction : char {
    Forward = 'F',   ///< Forward direction
    Backward = 'B',  ///< Backward direction
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(Direction, Forward, Backward)

namespace internal {
    struct Forward {
        constexpr operator Direction() const noexcept
        {
            return Direction::Forward;
        }
    };
    struct Backward {
        constexpr operator Direction() const noexcept
        {
            return Direction::Backward;
        }
    };
}  // namespace internal

// Constant expressions for directions

/// Forward direction
constexpr internal::Forward FORWARD{};
/// Backward direction
constexpr internal::Backward BACKWARD{};

// -----------------------------------------------------------------------------
// Storage types

enum class StoreV : char {
    Columnwise = 'C',  ///< Columnwise storage
    Rowwise = 'R',     ///< Rowwise storage
};
TLAPACK_DEF_OSTREAM_FOR_ENUM_WITH_2_VALUES(StoreV, Columnwise, Rowwise)

namespace internal {
    struct ColumnwiseStorage {
        constexpr operator StoreV() const noexcept
        {
            return StoreV::Columnwise;
        }
    };
    struct RowwiseStorage {
        constexpr operator StoreV() const noexcept { return StoreV::Rowwise; }
    };
}  // namespace internal

// Constant expressions for storage types

/// Columnwise storage
constexpr internal::ColumnwiseStorage COLUMNWISE_STORAGE{};
/// Rowwise storage
constexpr internal::RowwiseStorage ROWWISE_STORAGE{};

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
};

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
