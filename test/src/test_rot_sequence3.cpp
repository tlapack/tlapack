/// @file test_rot_sequence3.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test application of sequence of rotations
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
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/lapack/rot_kernel.hpp>
#include <tlapack/lapack/rot_sequence.hpp>
#include <tlapack/lapack/rot_sequence3.hpp>

#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE("Application of rotation sequence is accurate",
                   "[auxiliary]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using real_matrix_t = real_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;
    Create<real_matrix_t> new_real_matrix;

    // MatrixMarket reader
    MatrixMarket mm;
    rand_generator gen;

    const Side side = GENERATE(Side::Left, Side::Right);
    const Direction direction =
        GENERATE(Direction::Forward, Direction::Backward);
    const idx_t n = GENERATE(1, 2, 3, 4, 5, 10, 13);
    const idx_t m = GENERATE(1, 2, 3, 4, 5, 10, 13);
    const idx_t l = GENERATE(1, 2, 3, 4);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " l = " << l << " side = "
                           << side << " direction = " << direction)
    {
        const idx_t k = (side == Side::Left) ? m - 1 : n - 1;

        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(k) * eps;

        if (k < 1) SKIP_TEST;

        // Define the matrices and vectors
        std::vector<T> A_;
        auto A = new_matrix(A_, m, n);
        std::vector<T> B_;
        auto B = new_matrix(B_, m, n);
        std::vector<real_t> c_;
        auto C = new_real_matrix(c_, k, l);
        std::vector<T> s_;
        auto S = new_matrix(s_, k, l);

        mm.random(A);

        // Generate random rotation matrices
        for (idx_t j = 0; j < l; ++j) {
            for (idx_t i = 0; i < k; ++i) {
                T t1 = rand_helper<T>(gen);
                T t2 = rand_helper<T>(gen);
                rotg(t1, t2, C(i, j), S(i, j));
            }
        }
        tlapack::lacpy(GENERAL, A, B);

        // Apply the rotations
        rot_sequence3(side, direction, C, S, A);

        // Apply the rotations using rot_sequence
        for (idx_t j = 0; j < l; ++j) {
            auto c = col(C, j);
            auto s = col(S, j);
            rot_sequence(side, direction, c, s, B);
        }

        real_t bnorm = lange(MAX_NORM, B);
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = 0; i < m; ++i) {
                B(i, j) -= A(i, j);
            }
        }
        real_t res_norm = lange(MAX_NORM, B);

        CHECK(res_norm <= tol * bnorm);
    }
}

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
