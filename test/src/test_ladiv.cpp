/// @file test_ladiv.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <cfloat>

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Other routines
#include <tlapack/lapack/ladiv.hpp>

using namespace tlapack;

TEST_CASE("ladiv works properly", "[aux]")
{
    using real_t = double;

    const real_t M = std::numeric_limits<real_t>::max();
    const int E = std::numeric_limits<real_t>::max_exponent;
    const real_t eps = std::numeric_limits<real_t>::epsilon();

    real_t a, b, c, d, p, q;

    {
        a = b = c = d = pow(2, E - 1);
        ladiv(a, b, c, d, p, q);

        CHECK(p == 1.0);
        CHECK(q == 0.0);
    }
    {
        a = b = M;
        c = eps;
        d = 0;

        ladiv(a, b, c, d, p, q);

        INFO("p = " << p);
        INFO("q = " << q);

        CHECK(isinf(p));
        CHECK(isinf(q));
    }
    {
        a = b = M;
        c = 0;
        d = eps;

        ladiv(a, b, c, d, p, q);

        INFO("p = " << p);
        INFO("q = " << q);

        CHECK(isinf(p));
        CHECK(isinf(q));
    }
}