/// @file test_inv_house.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test generation of inverse householder reflector.
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>

// Other routines
#include <tlapack/lapack/inv_house3.hpp>
#include <tlapack/lapack/larf.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Inverse householder calculation is correct",
                   "[aux]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using TA = type_t<matrix_t>;
    using real_t = real_type<TA>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;
    const int seed = GENERATE(2, 3, 4, 5, 6, 7, 8, 9);

    const real_t tol = (real_t)5 * ulp<real_t>();

    // Seed random number generator
    mm.gen.seed(seed);

    // Define the matrices
    std::vector<TA> A_;
    auto A = new_matrix(A_, 2, 3);
    std::vector<TA> v(3);

    mm.random(A);

    DYNAMIC_SECTION("seed = " << seed)
    {
        TA tau;
        inv_house3(A, v, tau);

        // Apply reflector to A
        larf(Side::Right, Direction::Forward, StoreV::Columnwise, v, tau, A);

        auto anorm = lange(MAX_NORM, A);

        CHECK(abs(A(0, 0)) <= tol * anorm);
        CHECK(abs(A(1, 0)) <= tol * anorm);
    }
}
