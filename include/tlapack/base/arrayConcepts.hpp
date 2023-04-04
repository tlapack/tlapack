/// @file arrayConcepts.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ARRAY_CONCEPTS
#define TLAPACK_ARRAY_CONCEPTS

#include <cstddef>
#include <concepts>
#include <type_traits>

#include "tlapack/base/arrayTraits.hpp"

namespace tlapack {

#if __cplusplus >= 202002L

    // 
    // These concepts of matrices and vectors are not complete.
    // Currently, they are able to differentiate between a matrix, vector and number
    // but ideally, any function that satisfies the concept would work for the entire matrix,
    // i.e. slice, diag, ...
    // 


    template<typename A_t, typename idx_t=size_type<A_t>, typename T=type_t<A_t>>
    concept MatrixConcept = requires(A_t A, idx_t i, idx_t j)
    {
        { A( i, j ) } -> std::convertible_to<T>;
    };

    template<typename A_t, typename idx_t=size_type<A_t>, typename T=type_t<A_t>>
    concept VectorConcept = requires(A_t A, idx_t i)
    {
        { A[i] } -> std::convertible_to<T>;
    };

    template<typename T>
    concept NumberConcept = std::is_arithmetic<T>::value;

    #define AbstractMatrix MatrixConcept
    #define AbstractVector VectorConcept
    #define AbstractNumber NumberConcept
#else
    // Concepts are a C++20 feature, so just define them as `class` for earlier versions
    #define AbstractMatrix class
    #define AbstractVector class
    #define AbstractNumber class
#endif

}  // namespace tlapack

#endif  // TLAPACK_ARRAY_CONCEPTS