/// @file test_ladiv.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cfloat>

// Other routines
#include <tlapack/lapack/ladiv.hpp>

using namespace tlapack;

TEST_CASE("ladiv works properly", "[aux]")
{
    double a, b, c, d, p, q;

    a = b = c = d = pow(2, DBL_MAX_EXP - 1);
    ladiv(a, b, c, d, p, q);

    CHECK(p == 1.0);
    CHECK(q == 0.0);
}