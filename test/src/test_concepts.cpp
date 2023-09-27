/// @file test_concepts.cpp Test concepts in concepts.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Load NaNPropagComplex
#include "NaNPropagComplex.hpp"

#if __cplusplus >= 202002L

using namespace tlapack::concepts;

template <TLAPACK_COMPLEX T>
void f(T x)
{
    return;
}

TEST_CASE("Concept Arithmetic works as expected", "[concept]")
{
    REQUIRE(Arithmetic<int>);
    REQUIRE(Arithmetic<int&>);
    REQUIRE(Arithmetic<const int&> == false);
    REQUIRE(Arithmetic<int&&>);
    REQUIRE(Arithmetic<const int&&> == false);
}

TEST_CASE("Concept Vector works as expected", "[concept]")
{
    REQUIRE(Vector<std::vector<float>>);
}

TEST_CASE("Concept Complex works as expected", "[concept]")
{
    REQUIRE(Complex<std::complex<float>>);
    REQUIRE(Complex<tlapack::NaNPropagComplex<float>>);
}

TEST_CASE("Concept SliceableMatrix works as expected", "[concept]")
{
    #ifdef TLAPACK_TEST_EIGEN
    using matrix_t = Eigen::Matrix<std::complex<float>, -1, -1, 1, -1, -1>;
    REQUIRE(SliceableMatrix<matrix_t>);

    matrix_t A(2, 2);
    auto B =
        tlapack::slice(A, std::pair<int, int>{0, 1}, std::pair<int, int>{0, 1});
    REQUIRE(Matrix<decltype(B)>);

    B(0, 0);
    auto m = tlapack::nrows(B);
    auto n = tlapack::ncols(B);
    auto s = tlapack::size(B);
    REQUIRE(std::integral<decltype(m)>);
    REQUIRE(std::integral<decltype(n)>);
    REQUIRE(std::integral<decltype(s)>);
    #endif
}

#endif