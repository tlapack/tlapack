/// @file concepts.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ARRAY_CONCEPTS
#define TLAPACK_ARRAY_CONCEPTS

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/types.hpp"

#if __cplusplus >= 202002L
    #include <concepts>

namespace tlapack {

// Forward declarations
template <typename T>
inline T abs(const T& x);

namespace concepts {

    using std::ceil;
    using std::floor;
    using std::isinf;
    using std::isnan;
    using std::pow;
    using std::sqrt;

    template <typename matrix_t>
    concept Matrix = requires(matrix_t A)
    {
        // Entry access using operator()
        A(0, 0);

        // Number of rows
        {
            nrows(A)
        }
        ->std::convertible_to<int>;

        // Number of columns
        {
            ncols(A)
        }
        ->std::convertible_to<int>;
    };

    template <typename matrix_t>
    concept SliceableMatrix = Matrix<matrix_t>&& requires(matrix_t A)
    {
        // Submatrix view (matrix)
        slice(A, std::pair<int, int>{0, 1}, std::pair<int, int>{0, 1});

        // View of multiple rows (matrix)
        rows(A, std::pair<int, int>{0, 1});

        // View of multiple columns (matrix)
        cols(A, std::pair<int, int>{0, 1});

        // Row view (vector)
        row(A, 0);

        // Column view (vector)
        col(A, 0);

        // View of a slice of a row (vector)
        slice(A, 0, std::pair<int, int>{0, 1});

        // View of a slice of a column (vector)
        slice(A, std::pair<int, int>{0, 1}, 0);

        // Diagonal view (vector)
        diag(A);

        // Off-diagonal view (vector)
        diag(A, 1);
    };

    template <typename vector_t>
    concept Vector = requires(vector_t v)
    {
        // Entry access using operator[]
        v[0];

        // Number of entries
        {
            size(v)
        }
        ->std::convertible_to<int>;
    };

    template <typename vector_t>
    concept SliceableVector = Vector<vector_t>&& requires(vector_t v)
    {
        // Subvector view
        slice(v, std::pair<int, int>{0, 0});
    };

    template <typename T>
    concept Scalar = requires(T a, T b)
    {
        // Arithmetic operations
        a + b;
        a - b;
        a* b;
        a / b;

        // Absolute value
        abs(a);
    };

    template <typename T>
    concept Real = Scalar<T>&& requires(T a)
    {
        // Math functions
        sqrt(a);
        pow(a, 0);
        ceil(a);
        floor(a);

        // Inf check
        isinf(a);

        // NaN check
        isnan(a);
    };

    template <typename T>
    concept Complex = Scalar<T>&& requires(T a)
    {
        real(a);
        imag(a);
        conj(a);
    };

    template <typename T>
    concept Side = std::convertible_to<T, tlapack::Side>;

    template <typename T>
    concept Direction = std::convertible_to<T, tlapack::Direction>;

    template <typename T>
    concept Op = std::convertible_to<T, tlapack::Op>;

    template <typename T>
    concept StoreV = std::convertible_to<T, tlapack::StoreV>;

    template <typename T>
    concept Norm = std::convertible_to<T, tlapack::Norm>;

    template <typename T>
    concept Uplo = std::convertible_to<T, tlapack::Uplo>;

}  // namespace concepts
}  // namespace tlapack

    #define TLAPACK_MATRIX concepts::Matrix
    #define TLAPACK_SMATRIX concepts::SliceableMatrix

    #define TLAPACK_VECTOR concepts::Vector
    #define TLAPACK_SVECTOR concepts::SliceableVector

    #define TLAPACK_SCALAR concepts::Scalar
    #define TLAPACK_REAL concepts::Real
    #define TLAPACK_COMPLEX concepts::Complex

    #define TLAPACK_SIDE concepts::Side
    #define TLAPACK_DIRECTION concepts::Direction
    #define TLAPACK_OP concepts::Op
    #define TLAPACK_STOREV concepts::StoreV
    #define TLAPACK_NORM concepts::Norm
    #define TLAPACK_UPLO concepts::Uplo
#else
    // Concepts are a C++20 feature, so just define them as `class` for earlier
    // versions
    #define TLAPACK_MATRIX class
    #define TLAPACK_SMATRIX class

    #define TLAPACK_VECTOR class
    #define TLAPACK_SVECTOR class

    #define TLAPACK_SCALAR class
    #define TLAPACK_REAL class
    #define TLAPACK_COMPLEX class

    #define TLAPACK_SIDE class
    #define TLAPACK_DIRECTION class
    #define TLAPACK_OP class
    #define TLAPACK_STOREV class
    #define TLAPACK_NORM class
    #define TLAPACK_UPLO class

#endif

#endif  // TLAPACK_ARRAY_CONCEPTS