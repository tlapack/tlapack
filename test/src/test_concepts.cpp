/// @file test_concepts.cpp Test concepts in concepts.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <tlapack/base/concepts.hpp>

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

#endif