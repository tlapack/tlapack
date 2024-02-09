/// @file test_rot_kernel.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test kernels for applying rotations to a matrix
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/blas/nrm2.hpp>

// Other routines
#include <tlapack/lapack/rot_kernel.hpp>

#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE("rot_nofuse is accurate",
                   "[auxiliary]",
                   float,
                   double,
                   std::complex<float>,
                   std::complex<double>)
{
    using T = TestType;
    using real_t = real_type<T>;
    using idx_t = size_t;

    const idx_t n = GENERATE(1, 2, 4, 5, 8, 10, 16, 17);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(5. * n) * eps;

    DYNAMIC_SECTION("n = " << n)
    {
        // Define the vectors
        std::vector<T> x1(n);
        std::vector<T> x2(n);

        // Fill the vectors with random values
        rand_generator gen;
        for (idx_t i = 0; i < n; ++i) {
            x1[i] = rand_helper<T>(gen);
            x2[i] = rand_helper<T>(gen);
        }

        // Generate random rotation
        real_t t1 = rand_helper<real_t>(gen);
        real_t t2 = rand_helper<real_t>(gen);
        real_t c;
        real_t s;
        rotg(t1, t2, c, s);

        real_t norm1 = nrm2(x1);
        real_t norm2 = nrm2(x2);

        // Calculate reference solution
        std::vector<T> x1_ref = x1;
        std::vector<T> x2_ref = x2;
        rot(x1_ref, x2_ref, c, s);

        // Use the routine
        rot_nofuse(n, x1.data(), x2.data(), c, s);

        // Check the result
        for (idx_t i = 0; i < n; ++i) {
            x1_ref[i] -= x1[i];
            x2_ref[i] -= x2[i];
        }

        real_t error1 = nrm2(x1_ref);
        CHECK(error1 <= tol * norm1);
        real_t error2 = nrm2(x2_ref);
        CHECK(error2 <= tol * norm2);
    }
}

TEMPLATE_TEST_CASE("rot_fuse2x1 is accurate",
                   "[auxiliary]",
                   float,
                   double,
                   std::complex<float>,
                   std::complex<double>)
{
    using T = TestType;
    using real_t = real_type<T>;
    using idx_t = size_t;

    const idx_t n = GENERATE(1, 2, 4, 5, 8, 10, 16, 17);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(5. * n) * eps;

    DYNAMIC_SECTION("n = " << n)
    {
        // Define the vectors
        std::vector<T> x1(n);
        std::vector<T> x2(n);
        std::vector<T> x3(n);

        // Fill the vectors with random values
        rand_generator gen;
        for (idx_t i = 0; i < n; ++i) {
            x1[i] = rand_helper<T>(gen);
            x2[i] = rand_helper<T>(gen);
            x3[i] = rand_helper<T>(gen);
        }

        // Generate random rotation
        real_t t1, t2, c1, s1, c2, s2;
        t1 = rand_helper<real_t>(gen);
        t2 = rand_helper<real_t>(gen);
        rotg(t1, t2, c1, s1);
        t1 = rand_helper<real_t>(gen);
        t2 = rand_helper<real_t>(gen);
        rotg(t1, t2, c2, s2);

        real_t norm1 = nrm2(x1);
        real_t norm2 = nrm2(x2);
        real_t norm3 = nrm2(x3);

        // Calculate reference solution
        std::vector<T> x1_ref = x1;
        std::vector<T> x2_ref = x2;
        std::vector<T> x3_ref = x3;
        rot(x1_ref, x2_ref, c1, s1);
        rot(x2_ref, x3_ref, c2, s2);

        // Use the routine
        rot_fuse2x1(n, x1.data(), x2.data(), x3.data(), c1, s1, c2, s2);

        // Check the result
        for (idx_t i = 0; i < n; ++i) {
            x1_ref[i] -= x1[i];
            x2_ref[i] -= x2[i];
            x3_ref[i] -= x3[i];
        }

        real_t error1 = nrm2(x1_ref);
        CHECK(error1 <= tol * norm1);
        real_t error2 = nrm2(x2_ref);
        CHECK(error2 <= tol * norm2);
        real_t error3 = nrm2(x3_ref);
        CHECK(error3 <= tol * norm3);
    }
}

TEMPLATE_TEST_CASE("rot_fuse1x2 is accurate",
                   "[auxiliary]",
                   float,
                   double,
                   std::complex<float>,
                   std::complex<double>)
{
    using T = TestType;
    using real_t = real_type<T>;
    using idx_t = size_t;

    const idx_t n = GENERATE(1, 2, 4, 5, 8, 10, 16, 17);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(5. * n) * eps;

    DYNAMIC_SECTION("n = " << n)
    {
        // Define the vectors
        std::vector<T> x1(n);
        std::vector<T> x2(n);
        std::vector<T> x3(n);

        // Fill the vectors with random values
        rand_generator gen;
        for (idx_t i = 0; i < n; ++i) {
            x1[i] = rand_helper<T>(gen);
            x2[i] = rand_helper<T>(gen);
            x3[i] = rand_helper<T>(gen);
        }

        // Generate random rotation
        real_t t1, t2, c1, s1, c2, s2;
        t1 = rand_helper<real_t>(gen);
        t2 = rand_helper<real_t>(gen);
        rotg(t1, t2, c1, s1);
        t1 = rand_helper<real_t>(gen);
        t2 = rand_helper<real_t>(gen);
        rotg(t1, t2, c2, s2);

        real_t norm1 = nrm2(x1);
        real_t norm2 = nrm2(x2);
        real_t norm3 = nrm2(x3);

        // Calculate reference solution
        std::vector<T> x1_ref = x1;
        std::vector<T> x2_ref = x2;
        std::vector<T> x3_ref = x3;
        rot(x2_ref, x3_ref, c1, s1);
        rot(x1_ref, x2_ref, c2, s2);

        // Use the routine
        rot_fuse1x2(n, x1.data(), x2.data(), x3.data(), c1, s1, c2, s2);

        // Check the result
        for (idx_t i = 0; i < n; ++i) {
            x1_ref[i] -= x1[i];
            x2_ref[i] -= x2[i];
            x3_ref[i] -= x3[i];
        }

        real_t error1 = nrm2(x1_ref);
        CHECK(error1 <= tol * norm1);
        real_t error2 = nrm2(x2_ref);
        CHECK(error2 <= tol * norm2);
        real_t error3 = nrm2(x3_ref);
        CHECK(error3 <= tol * norm3);
    }
}

TEMPLATE_TEST_CASE("rot_fuse2x2 is accurate",
                   "[auxiliary]",
                   float,
                   double,
                   std::complex<float>,
                   std::complex<double>)
{
    using T = TestType;
    using real_t = real_type<T>;
    using idx_t = size_t;

    const idx_t n = GENERATE(1, 2, 4, 5, 8, 10, 16, 17);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(5. * n) * eps;

    DYNAMIC_SECTION("n = " << n)
    {
        // Define the vectors
        std::vector<T> x1(n);
        std::vector<T> x2(n);
        std::vector<T> x3(n);
        std::vector<T> x4(n);

        // Fill the vectors with random values
        rand_generator gen;
        for (idx_t i = 0; i < n; ++i) {
            x1[i] = rand_helper<T>(gen);
            x2[i] = rand_helper<T>(gen);
            x3[i] = rand_helper<T>(gen);
            x4[i] = rand_helper<T>(gen);
        }

        // Generate random rotation
        real_t t1, t2, c1, s1, c2, s2, c3, s3, c4, s4;
        t1 = rand_helper<real_t>(gen);
        t2 = rand_helper<real_t>(gen);
        rotg(t1, t2, c1, s1);
        t1 = rand_helper<real_t>(gen);
        t2 = rand_helper<real_t>(gen);
        rotg(t1, t2, c2, s2);
        t1 = rand_helper<real_t>(gen);
        t2 = rand_helper<real_t>(gen);
        rotg(t1, t2, c3, s3);
        t1 = rand_helper<real_t>(gen);
        t2 = rand_helper<real_t>(gen);
        rotg(t1, t2, c4, s4);

        real_t norm1 = nrm2(x1);
        real_t norm2 = nrm2(x2);
        real_t norm3 = nrm2(x3);
        real_t norm4 = nrm2(x4);

        // Calculate reference solution
        std::vector<T> x1_ref = x1;
        std::vector<T> x2_ref = x2;
        std::vector<T> x3_ref = x3;
        std::vector<T> x4_ref = x4;
        rot(x2_ref, x3_ref, c1, s1);
        rot(x1_ref, x2_ref, c2, s2);
        rot(x3_ref, x4_ref, c3, s3);
        rot(x2_ref, x3_ref, c4, s4);

        // Use the routine
        rot_fuse2x2(n, x1.data(), x2.data(), x3.data(), x4.data(), c1, s1, c2,
                    s2, c3, s3, c4, s4);

        // Check the result
        for (idx_t i = 0; i < n; ++i) {
            x1_ref[i] -= x1[i];
            x2_ref[i] -= x2[i];
            x3_ref[i] -= x3[i];
            x4_ref[i] -= x4[i];
        }

        real_t error1 = nrm2(x1_ref);
        CHECK(error1 <= tol * norm1);
        real_t error2 = nrm2(x2_ref);
        CHECK(error2 <= tol * norm2);
        real_t error3 = nrm2(x3_ref);
        CHECK(error3 <= tol * norm3);
        real_t error4 = nrm2(x4_ref);
        CHECK(error4 <= tol * norm4);
    }
}