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

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/types.hpp"

namespace tlapack {

#if __cplusplus >= 202002L

//
// These concepts of matrices and vectors are not complete.
// Currently, they are able to differentiate between a matrix, vector and number
// but ideally, any function that satisfies the concept would work for the
// entire matrix, i.e. slice, diag, ...
//

template <typename A_t,
          typename idx_t = size_type<A_t>,
          typename T = type_t<A_t>,
          typename pair_t = std::pair<idx_t, idx_t>>
concept MatrixConcept = requires(A_t A, idx_t i, idx_t j, pair_t pi, pair_t pj)
{
    {
        A(i, j)
    }
    ->std::convertible_to<T>;
    {
        nrows(A)
    }
    ->std::convertible_to<idx_t>;
    {
        ncols(A)
    }
    ->std::convertible_to<idx_t>;
    // row(A, i);
    // col(A, j);
    // rows(A, pi);
    // cols(A, pj);
    // diag(A);
    // slice( A, pi, pj );
    // slice( A, i, pj );
    // slice( A, pi, j );
};

template <typename V_t,
          typename idx_t = size_type<V_t>,
          typename T = type_t<V_t>,
          typename pair_t = std::pair<idx_t, idx_t>>
concept VectorConcept = requires(V_t v, idx_t i, pair_t pi)
{
    {
        v[i]
    }
    ->std::convertible_to<T>;
    {
        size(v)
    }
    ->std::convertible_to<idx_t>;
    // slice(v, pi);
};

// We use our own concept here instead of std::is_arithmetic,
// because is_arithmetic does not include complex numbers
template <typename T>
concept NumberConcept = requires(T a, T b)
{
    a + b;
    a* b;
};

template <typename T>
concept SideConcept = std::is_convertible<T, Side>::value;

template <typename T>
concept DirectionConcept = std::is_convertible<T, Direction>::value;

template <typename T>
concept OpConcept = std::is_convertible<T, Op>::value;

template <typename T>
concept StoreVConcept = std::is_convertible<T, StoreV>::value;

template <typename T>
concept NormConcept = std::is_convertible<T, Norm>::value;

template <typename T>
concept UploConcept = std::is_convertible<T, Uplo>::value;

    #define TLAPACK_MATRIX MatrixConcept
    #define TLAPACK_VECTOR VectorConcept
    #define TLAPACK_NUMBER NumberConcept
    #define TLAPACK_SIDE SideConcept
    #define TLAPACK_DIRECTION DirectionConcept
    #define TLAPACK_OP OpConcept
    #define TLAPACK_STOREV StoreVConcept
    #define TLAPACK_NORM NormConcept
    #define TLAPACK_UPLO UploConcept
#else
    // Concepts are a C++20 feature, so just define them as `class` for earlier
    // versions
    #define TLAPACK_MATRIX class
    #define TLAPACK_VECTOR class
    #define TLAPACK_NUMBER class
    #define TLAPACK_DIRECTION class
    #define TLAPACK_OP class
    #define TLAPACK_STOREV class
    #define TLAPACK_NORM class
    #define TLAPACK_UPLO class
#endif

}  // namespace tlapack

#endif  // TLAPACK_ARRAY_CONCEPTS