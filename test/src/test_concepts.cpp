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

using namespace tlapack::concepts;

template <Vector T>
void test_vector()
{}

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
    test_vector<std::vector<float>>();
    REQUIRE(Vector<std::vector<float>>);
}