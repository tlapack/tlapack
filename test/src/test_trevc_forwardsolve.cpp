/// @file test_trevc_forwardsolve.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test eigenvector calculations.
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include <string>

#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/blas/iamax.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>

// Other routines
#include <tlapack/blas/gemv.hpp>
#include <tlapack/lapack/trevc_forwardsolve.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("TREVC_forwardsolve correctly computes the left eigenvector",
                   "[eigenvalues][eigenvectors][trevc]",
                   TLAPACK_LEGACY_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<TA>;
    using complex_t = complex_type<real_t>;
    using range = pair<idx_t, idx_t>;
    using vector_t = vector_type<TestType>;

    // Eigenvalues and 16-bit types do not mix well
    if constexpr (sizeof(real_t) <= 2) SKIP_TEST;

    // Functor
    Create<matrix_t> new_matrix;
    Create<vector_t> new_vector;

    // MatrixMarket reader
    MatrixMarket mm;

    const int seed = GENERATE(2, 3);

    const idx_t n = GENERATE(1, 2, 3, 4, 5, 8, 10);
    const real_t zero(0);
    const real_t one(1);

    // Seed random number generator
    mm.gen.seed(seed);

    // Define the matrices
    std::vector<TA> T_;
    auto T = new_matrix(T_, n, n);

    mm.random(Uplo::Upper, T);
    // Set lower triangle to zero
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            T(i, j) = TA(zero);

    // Randomly set some subdiagonal entries to non-zero to create 2x2 blocks
    if constexpr (is_real<TA>) {
        idx_t j = 0;
        while (j + 1 < n) {
            if (rand_helper<float>(mm.gen) < 0.5f) {
                // Generate a 2x2 block in normalized form
                TA alpha = rand_helper<TA>(mm.gen);
                TA beta = rand_helper<TA>(mm.gen);
                TA gamma = rand_helper<TA>(mm.gen);
                T(j, j) = alpha;
                T(j, j + 1) = beta;
                T(j + 1, j) = -gamma;
                T(j + 1, j + 1) = alpha;
                j += 2;
            }
            else {
                j += 1;
            }
        }
    }

    // Precompute column norms for scaling
    std::vector<real_t> colN_(n);
    auto colN = new_vector(colN_, n);
    trevc_colnorms(Norm::One, T, colN);

    for (idx_t k = 0; k < n; ++k) {
        DYNAMIC_SECTION(" n = " << n << " seed = " << seed << " k = " << k)
        {
            if (k > 0) {
                if (T(k, k - 1) != TA(zero)) {
                    // Skip the second value of a 2x2 block
                    continue;
                }
            }

            bool is_2x2_block = false;
            if (k + 1 < n) {
                if (T(k + 1, k) != TA(zero)) {
                    is_2x2_block = true;
                }
            }

            if (is_2x2_block) {
                if constexpr (is_real<TA>) {
                    //
                    // Compute left eigenvector using trevc3_forwardsolve_double
                    //
                    std::vector<TA> v_real_;
                    auto v_real = new_vector(v_real_, n);
                    std::vector<TA> v_imag_;
                    auto v_imag = new_vector(v_imag_, n);

                    trevc_forwardsolve_double(T, v_real, v_imag, k, colN);

                    std::vector<complex_t> v_;
                    auto v = new_vector(v_, n);
                    for (idx_t i = 0; i < n; ++i) {
                        v[i] = complex_t(v_real[i], v_imag[i]);
                    }

                    // Check that v_ is nonzero
                    real_t normv = asum(v);
                    REQUIRE(normv != real_t(0));

                    //
                    // Verify that v_**H * T = lambda*v_**H
                    // (or equivalently T**H * v_ = conj(lambda)*v_)
                    //

                    TA alpha = T(k, k);
                    TA beta = T(k, k + 1);
                    TA gamma = T(k + 1, k);
                    // eigenvalue
                    TA lambda_real = alpha;
                    TA lambda_imag = sqrt(abs(beta)) * sqrt(abs(gamma));

                    complex_t lambda(lambda_real, lambda_imag);

                    std::vector<complex_t> Tv_;
                    auto Tv = new_vector(Tv_, n);

                    gemv(Op::ConjTrans, one, T, v_, zero, Tv);

                    real_t tol = ulp<real_t>() * normv * real_t(n);
                    for (idx_t i = 0; i < n; ++i) {
                        CHECK(abs(Tv[i] - conj(lambda) * v[i]) <= tol);
                    }
                }
            }
            else {
                std::vector<TA> v_;
                auto v = new_vector(v_, n);
                //
                // Compute left eigenvector using trevc3_forwardsolve_single
                //
                trevc_forwardsolve_single(T, v, k, colN);

                // Check that v is nonzero
                real_t normv = asum(v);
                REQUIRE(normv != real_t(0));

                //
                // Verify that v**H * T = lambda*v**H
                // (or equivalently T**H * v = conj(lambda)*v)
                //
                TA lambda = T(k, k);
                std::vector<TA> Tv_;
                auto Tv = new_vector(Tv_, n);
                gemv(Op::ConjTrans, one, T, v, zero, Tv);

                real_t tol = ulp<real_t>() * normv * real_t(n);
                for (idx_t i = 0; i < n; ++i) {
                    CHECK(abs(Tv[i] - conj(lambda) * v[i]) <= tol);
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "TREVC_forwardsolve correctly computes the left eigenvector with scaling",
    "[eigenvalues][eigenvectors][trevc]",
    TLAPACK_LEGACY_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<TA>;
    using complex_t = complex_type<real_t>;
    using range = pair<idx_t, idx_t>;
    using vector_t = vector_type<TestType>;

    // Eigenvalues and 16-bit types do not mix well
    if constexpr (sizeof(real_t) <= 2) SKIP_TEST;

    // Functor
    Create<matrix_t> new_matrix;
    Create<vector_t> new_vector;

    // MatrixMarket reader
    MatrixMarket mm;

    const int seed = GENERATE(2);
    const real_t zero(0);
    const real_t one(1);

    // Seed random number generator
    mm.gen.seed(seed);

    // Define the matrices
    idx_t n = GENERATE(20, 40);

    const TA a = 1.0e6 + 1;
    const TA b = 1;
    const TA c = 1.0e6;

    // const TA a = 0;
    // const TA b = -1;
    // const TA c = 0.5;

    // Define the matrices
    std::vector<TA> T_;
    auto T = new_matrix(T_, n, n);

    // Matrix as described in section 3 of "Robust parallel eigenvector
    // computation for the non-symmetric eigenvalue problem"
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < j; ++i)
            T(i, j) = -c;
        T(j, j) = a - b * TA(j);
        for (idx_t i = j + 1; i < n; ++i)
            T(i, j) = TA(zero);
    }

    // Randomly set some subdiagonal entries to non-zero to create 2x2 blocks
    if constexpr (is_real<TA>) {
        idx_t j = 0;
        while (j + 1 < n) {
            if (rand_helper<float>(mm.gen) < 0.8f) {
                T(j + 1, j) = c;
                T(j + 1, j + 1) = T(j, j);
                j += 2;
            }
            else {
                j += 1;
            }
        }
    }

    // Precompute column norms for scaling
    std::vector<real_t> colN_(n);
    auto colN = new_vector(colN_, n);
    trevc_colnorms(Norm::One, T, colN);

    for (idx_t k = 0; k < n; ++k) {
        DYNAMIC_SECTION(" n = " << n << " seed = " << seed << " k = " << k)
        {
            if (k > 0) {
                if (T(k, k - 1) != TA(zero)) {
                    // Skip the second value of a 2x2 block
                    continue;
                }
            }

            bool is_2x2_block = false;
            if (k + 1 < n) {
                if (T(k + 1, k) != TA(zero)) {
                    is_2x2_block = true;
                }
            }

            if (is_2x2_block) {
                if constexpr (is_real<TA>) {
                    //
                    // Compute left eigenvector using trevc3_forwardsolve_double
                    //
                    std::vector<TA> v_real_;
                    auto v_real = new_vector(v_real_, n);
                    std::vector<TA> v_imag_;
                    auto v_imag = new_vector(v_imag_, n);

                    trevc_forwardsolve_double(T, v_real, v_imag, k, colN);

                    std::vector<complex_t> v_;
                    auto v = new_vector(v_, n);
                    for (idx_t i = 0; i < n; ++i) {
                        v[i] = complex_t(v_real[i], v_imag[i]);
                    }

                    // Check that v is finite
                    for (idx_t i = 0; i < n; ++i) {
                        REQUIRE(std::isfinite(real(v[i])));
                        REQUIRE(std::isfinite(imag(v[i])));
                    }

                    // Check that v_ is nonzero
                    real_t normv = asum(v);
                    REQUIRE(normv != real_t(0));
                    REQUIRE(!std::isnan(normv));

                    // Normalize v to avoid overflow in the next step
                    for (idx_t i = 0; i < n; ++i) {
                        v[i] /= normv;
                    }

                    //
                    // Verify that v_**H * T = lambda*v_**H
                    // (or equivalently T**H * v_ = conj(lambda)*v_)
                    //

                    TA alpha = T(k, k);
                    TA beta = T(k, k + 1);
                    TA gamma = T(k + 1, k);
                    // eigenvalue
                    TA lambda_real = alpha;
                    TA lambda_imag = sqrt(abs(beta)) * sqrt(abs(gamma));

                    complex_t lambda(lambda_real, lambda_imag);

                    std::vector<complex_t> Tv_;
                    auto Tv = new_vector(Tv_, n);

                    gemv(Op::ConjTrans, one, T, v_, zero, Tv);

                    real_t tol =
                        ulp<real_t>() * abs1(lambda) * real_t(n) * real_t(100);
                    for (idx_t i = 0; i < n; ++i) {
                        CHECK(abs(Tv[i] - conj(lambda) * v[i]) <= tol);
                    }
                }
            }
            else {
                std::vector<TA> v_;
                auto v = new_vector(v_, n);
                //
                // Compute left eigenvector using trevc3_forwardsolve_single
                //
                trevc_forwardsolve_single(T, v, k, colN);

                // Check that v is finite
                for (idx_t i = 0; i < n; ++i) {
                    REQUIRE(std::isfinite(real(v[i])));
                    REQUIRE(std::isfinite(imag(v[i])));
                }

                // Check that v is nonzero
                real_t normv = asum(v);
                REQUIRE(normv != real_t(0));
                REQUIRE(!std::isnan(normv));

                // normalize v to avoid overflow in the next step
                for (idx_t i = 0; i < n; ++i) {
                    v[i] /= normv;
                }

                //
                // Verify that v**H * T = lambda*v**H
                // (or equivalently T**H * v = conj(lambda)*v)
                //
                TA lambda = T(k, k);
                std::vector<TA> Tv_;
                auto Tv = new_vector(Tv_, n);
                gemv(Op::ConjTrans, one, T, v, zero, Tv);

                std::vector<TA> lambda_v_;
                auto lambda_v = new_vector(lambda_v_, n);
                for (idx_t i = 0; i < n; ++i) {
                    lambda_v[i] = conj(lambda) * v[i];
                }

                real_t tol =
                    ulp<real_t>() * real_t(n) * abs1(lambda) * real_t(100);
                for (idx_t i = 0; i < n; ++i) {
                    CHECK(abs(Tv[i] - conj(lambda) * v[i]) <= tol);
                }
            }
        }
    }
}
