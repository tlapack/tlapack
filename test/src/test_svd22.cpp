/// @file test_svd22.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test auxiliary routines that calculate svd of 2x2 triangular matrix
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Other routines
#include <tlapack/blas/rot.hpp>
#include <tlapack/lapack/singularvalues22.hpp>
#include <tlapack/lapack/svd22.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("2x2 svd gives correct result",
                   "[svd]",
                   TLAPACK_REAL_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    const T eps = uroundoff<T>();
    const T tol = T(4.0e1) * eps;

    const std::string matrix_type = GENERATE(
        "rand", "large f", "large g", "large h", "zero f", "zero g", "zero h");

    T f, g, h, ssmin1, ssmax1, ssmin2, ssmax2, csl, snl, csr, snr;

    if (matrix_type == "rand") {
        f = rand_helper<T>();
        g = rand_helper<T>();
        h = rand_helper<T>();
    }
    if (matrix_type == "large f") {
        f = T(100.) * rand_helper<T>();
        g = rand_helper<T>();
        h = rand_helper<T>();
    }
    if (matrix_type == "large g") {
        f = rand_helper<T>();
        g = T(100.) * rand_helper<T>();
        h = rand_helper<T>();
    }
    if (matrix_type == "large h") {
        f = rand_helper<T>();
        g = rand_helper<T>();
        h = T(100.) * rand_helper<T>();
    }
    if (matrix_type == "zero f") {
        f = T(0.);
        g = rand_helper<T>();
        h = rand_helper<T>();
    }
    if (matrix_type == "zero g") {
        f = rand_helper<T>();
        g = T(0.);
        h = rand_helper<T>();
    }
    if (matrix_type == "zero h") {
        f = rand_helper<T>();
        g = rand_helper<T>();
        h = T(0.);
    }

    DYNAMIC_SECTION("matrix type = " << matrix_type)
    {
        svd22(f, g, h, ssmin1, ssmax1, csl, snl, csr, snr);

        // Check the decomposition
        std::vector<T> A_;
        auto A = new_matrix(A_, 2, 2);
        A(0, 0) = f;
        A(0, 1) = g;
        A(1, 1) = h;
        A(1, 0) = T(0);
        auto r0 = row(A, 0);
        auto r1 = row(A, 1);
        rot(r0, r1, csl, snl);
        auto c0 = col(A, 0);
        auto c1 = col(A, 1);
        rot(c0, c1, csr, snr);
        CHECK(tlapack::abs(A(0, 0) - ssmax1) <= tol * tlapack::abs(ssmax1));
        CHECK(tlapack::abs(A(1, 1) - ssmin1) <= tol * tlapack::abs(ssmax1));
        CHECK(tlapack::abs(A(1, 0)) <= tol * tlapack::abs(ssmax1));
        CHECK(tlapack::abs(A(0, 1)) <= tol * tlapack::abs(ssmax1));

        // Check that the singular values calculated by svd22 and
        // singularvalues22 are the same
        singularvalues22(f, g, h, ssmin2, ssmax2);
        CHECK(tlapack::abs(tlapack::abs(ssmin1) - ssmin2) <=
              tol * tlapack::abs(ssmin1));
        CHECK(tlapack::abs(tlapack::abs(ssmax1) - ssmax2) <=
              tol * tlapack::abs(ssmax1));
    }
}
