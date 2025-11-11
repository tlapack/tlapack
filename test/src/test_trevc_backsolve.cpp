/// @file test_trevc_backsolve.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test eigenvector calculations.
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
#include <tlapack/blas/gemv.hpp>
#include <tlapack/lapack/trevc_backsolve.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("TREVC_backsolve correctly computes the right eigenvector",
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
            if (rand_helper<float>(mm.gen) < 0.8f) {
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
    for (idx_t j = 0; j < n; ++j) {
        idx_t itmax = iamax(slice(col(T, j), range(0, n)));
        colN[j] = abs1(T(itmax, j));
    }

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
                    // Compute right eigenvector using trevc3_backsolve_double
                    //
                    std::vector<TA> v_real_;
                    auto v_real = new_vector(v_real_, n);
                    std::vector<TA> v_imag_;
                    auto v_imag = new_vector(v_imag_, n);

                    trevc_backsolve_double(T, v_real, v_imag, k, colN);

                    // Check that v_real + i*v_imag is nonzero
                    real_t normv = asum(v_real) + asum(v_imag);
                    REQUIRE(normv != real_t(0));

                    //
                    // Verify that T*(v_real + i*v_imag) = lambda*(v_real +
                    // i*v_imag)
                    //

                    TA alpha = T(k, k);
                    TA beta = T(k, k + 1);
                    TA gamma = T(k + 1, k);
                    // eigenvalue
                    TA lambda_real = alpha;
                    TA lambda_imag = sqrt(abs(beta)) * sqrt(abs(gamma));

                    std::vector<TA> Tv_real_;
                    auto Tv_real = new_vector(Tv_real_, n);
                    std::vector<TA> Tv_imag_;
                    auto Tv_imag = new_vector(Tv_imag_, n);
                    gemv(Op::NoTrans, one, T, v_real, zero, Tv_real);
                    gemv(Op::NoTrans, one, T, v_imag, zero, Tv_imag);

                    real_t tol = ulp<real_t>() * normv * real_t(n);

                    std::vector<TA> v_real2_;
                    auto v_real2 = new_vector(v_real2_, n);
                    std::vector<TA> v_imag2_;
                    auto v_imag2 = new_vector(v_imag2_, n);

                    for (idx_t i = 0; i < n; ++i) {
                        // Compute lambda * (v_real + i * v_imag)
                        v_real2[i] =
                            lambda_real * v_real[i] - lambda_imag * v_imag[i];
                        v_imag2[i] =
                            lambda_real * v_imag[i] + lambda_imag * v_real[i];
                    }

                    for (idx_t i = 0; i < n; ++i) {
                        // Real part
                        CHECK(abs(Tv_real[i] - (lambda_real * v_real[i] -
                                                lambda_imag * v_imag[i])) <=
                              tol);
                        // Imaginary part
                        CHECK(abs(Tv_imag[i] - (lambda_real * v_imag[i] +
                                                lambda_imag * v_real[i])) <=
                              tol);
                    }
                }
            }
            else {
                std::vector<TA> v_;
                auto v = new_vector(v_, n);
                //
                // Compute right eigenvector using trevc3_backsolve_single
                //
                trevc_backsolve_single(T, v, k, colN);

                // Check that v is nonzero
                real_t normv = asum(v);
                REQUIRE(normv != real_t(0));

                //
                // Verify that T*v = lambda*v
                //
                TA lambda = T(k, k);
                std::vector<TA> Tv_;
                auto Tv = new_vector(Tv_, n);
                gemv(Op::NoTrans, one, T, v, zero, Tv);

                real_t tol = ulp<real_t>() * normv * real_t(n);
                for (idx_t i = 0; i < n; ++i) {
                    CHECK(abs(Tv[i] - lambda * v[i]) <= tol);
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "TREVC_backsolve correctly computes the right eigenvector with scaling",
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

    const TA zero(0);
    const TA one(1);

    // Seed random number generator
    mm.gen.seed(seed);

    idx_t n = 20;

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
    for (idx_t j = 0; j < n; ++j) {
        idx_t itmax = iamax(slice(col(T, j), range(0, n)));
        colN[j] = abs1(T(itmax, j));
    }

    for (idx_t k = 0; k < n; ++k) {
        DYNAMIC_SECTION(" n = " << n << " k = " << k)
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
                    // Compute right eigenvector using trevc3_backsolve_double
                    //
                    std::vector<TA> v_real_;
                    auto v_real = new_vector(v_real_, n);
                    std::vector<TA> v_imag_;
                    auto v_imag = new_vector(v_imag_, n);

                    trevc_backsolve_double(T, v_real, v_imag, k, colN);

                    // Check that v_real + i*v_imag is finite
                    for (idx_t i = 0; i < n; ++i) {
                        REQUIRE(std::isfinite(v_real[i]));
                        REQUIRE(std::isfinite(v_imag[i]));
                    }

                    // Check that v_real + i*v_imag is nonzero
                    real_t normv = asum(v_real) + asum(v_imag);
                    REQUIRE(normv != real_t(0));
                    REQUIRE(!std::isnan(normv));

                    // normalize v to avoid overflow in the next step
                    for (idx_t i = 0; i < n; ++i) {
                        v_real[i] /= normv;
                        v_imag[i] /= normv;
                    }

                    //
                    // Verify that T*(v_real + i*v_imag) = lambda*(v_real +
                    // i*v_imag)
                    //

                    TA alpha = T(k, k);
                    TA beta = T(k, k + 1);
                    TA gamma = T(k + 1, k);
                    // eigenvalue
                    TA lambda_real = alpha;
                    TA lambda_imag = sqrt(abs(beta)) * sqrt(abs(gamma));

                    std::vector<TA> Tv_real_;
                    auto Tv_real = new_vector(Tv_real_, n);
                    std::vector<TA> Tv_imag_;
                    auto Tv_imag = new_vector(Tv_imag_, n);
                    gemv(Op::NoTrans, one, T, v_real, zero, Tv_real);
                    gemv(Op::NoTrans, one, T, v_imag, zero, Tv_imag);

                    real_t tol = ulp<real_t>() *
                                 (abs(lambda_real) + abs(lambda_imag)) *
                                 real_t(n) * real_t(100);

                    std::vector<TA> v_real2_;
                    auto v_real2 = new_vector(v_real2_, n);
                    std::vector<TA> v_imag2_;
                    auto v_imag2 = new_vector(v_imag2_, n);

                    for (idx_t i = 0; i < n; ++i) {
                        // Compute lambda * (v_real + i * v_imag)
                        v_real2[i] =
                            lambda_real * v_real[i] - lambda_imag * v_imag[i];
                        v_imag2[i] =
                            lambda_real * v_imag[i] + lambda_imag * v_real[i];
                    }

                    for (idx_t i = 0; i < n; ++i) {
                        // Real part
                        CHECK(abs(Tv_real[i] - (lambda_real * v_real[i] -
                                                lambda_imag * v_imag[i])) <=
                              tol);
                        // Imaginary part
                        CHECK(abs(Tv_imag[i] - (lambda_real * v_imag[i] +
                                                lambda_imag * v_real[i])) <=
                              tol);
                    }
                }
            }
            else {
                std::vector<TA> v_;
                auto v = new_vector(v_, n);
                //
                // Compute right eigenvector using trevc3_backsolve_single
                //
                trevc_backsolve_single(T, v, k, colN);

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
                // Verify that T*v = lambda*v
                //
                TA lambda = T(k, k);
                std::vector<TA> Tv_;
                auto Tv = new_vector(Tv_, n);
                gemv(Op::NoTrans, one, T, v, zero, Tv);

                std::vector<TA> lambda_v_;
                auto lambda_v = new_vector(lambda_v_, n);
                for (idx_t i = 0; i < n; ++i) {
                    lambda_v[i] = lambda * v[i];
                }

                real_t tol =
                    ulp<real_t>() * abs(lambda) * real_t(n) * real_t(10);
                for (idx_t i = 0; i < n; ++i) {
                    CHECK(abs(Tv[i] - lambda * v[i]) <= tol);
                }
            }
        }
    }
}
