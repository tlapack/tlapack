/// @file test_larf.cpp Test the operations with Householder reflectors
/// @author Weslley S Pereira, University of Colorado Denver, USA
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

// Auxiliary routines
#include <tlapack/blas/dot.hpp>
#include <tlapack/blas/nrm2.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/lapack/larf.hpp>
#include <tlapack/lapack/larfg.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Generation of Householder reflectors",
                   "[larfg]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);
    using vector_t = vector_type<TestType>;
    using T = type_t<vector_t>;
    using idx_t = size_type<vector_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    // Functor
    Create<vector_t> new_vector;

    // Test parameters
    const idx_t n = GENERATE(10, 19, 30);
    const Direction direction =
        GENERATE(Direction::Forward, Direction::Backward);
    const StoreV storeMode = GENERATE(StoreV::Columnwise, StoreV::Rowwise);
    const std::string initialX = GENERATE(as<std::string>{}, "Zeros", "Random");
    const std::string typeAlpha =
        GENERATE(as<std::string>{}, "Real", "Complex");
    const std::string whichLARFG =
        GENERATE(as<std::string>{}, "direction,v", "alpha,x");

    // Skip tests with invalid parameters
    if (typeAlpha == "Complex" && is_real<T>) return;

    // Vectors
    std::vector<T> v_;
    auto v = new_vector(v_, n);
    std::vector<T> w_;
    auto w = new_vector(w_, n);

    // Constants
    const real_t tol = real_t(n) * ulp<real_t>();
    const real_t zero(0);
    const real_t one(1);
    const real_t two(2);
    const T alpha =
        (typeAlpha == "Real") ? rand_helper<real_t>() : rand_helper<T>();
    const idx_t alphaIdx = (direction == Direction::Forward) ? 0 : n - 1;

    // Print test parameters
    INFO("Which larfg: " << whichLARFG);
    INFO("Vector size: " << n);
    INFO("Direction: " << direction);
    INFO("Store mode: " << storeMode);
    INFO("Initial x: " << initialX);
    INFO("Type of alpha: " << typeAlpha);
    INFO("alpha = " << alpha);

    DYNAMIC_SECTION("n = " << n << " direction = " << direction
                           << " storeMode = " << storeMode << " initialX = "
                           << initialX << " typeAlpha = " << typeAlpha
                           << " whichLARFG = " << whichLARFG)
    {
        // Initialize v
        if (initialX == "Zeros") {
            for (idx_t i = 0; i < n; ++i) {
                v[i] = zero;
            }
        }
        else {  // initialX == 'R'
            for (idx_t i = 0; i < n; ++i) {
                v[i] = rand_helper<T>();
            }
        }
        v[alphaIdx] = alpha;

        // Copy v to w
        for (idx_t i = 0; i < n; ++i) {
            w[i] = v[i];
        }

        // Place to store the scalar factor of the Householder reflector
        T tau;

        if (whichLARFG == "alpha,x") {
            auto x =
                slice(v, (direction == Direction::Forward) ? range(1, n)
                                                           : range(0, n - 1));
            larfg(storeMode, v[alphaIdx], x, tau);
        }
        else  // whichLARFG == "direction,v"
        {
            larfg(direction, storeMode, v, tau);
        }

        // Post-process v and extract beta
        const T beta = v[alphaIdx];
        v[alphaIdx] = one;

        // Check that the imaginary part of beta is zero
        CHECK((imag(beta) == zero));

        // If the elements of x are all zero and alpha is real, then tau = 0
        if (typeAlpha == "Real" && initialX == "Zeros") {
            CHECK(tau == zero);
        }
        // Otherwise  1 <= real(tau) <= 2 and abs(tau-1) <= 1.
        else {
            CHECK(one <= real(tau));
            CHECK(real(tau) <= two);
            CHECK(tlapack::abs(tau - one) <= one);
        }

        const T vHw = dot(v, w);
        if (storeMode == StoreV::Columnwise) {
            // Check that (id - conj(tau)*v*v^H)*[alpha x]^t = [beta 0]^t
            for (idx_t i = 0; i < n; ++i)
                w[i] -= v[i] * conj(tau) * vHw;
        }
        else {
            // Check that [alpha x]*(id - tau*v*v^H) = [beta 0]
            for (idx_t i = 0; i < n; ++i)
                w[i] -= vHw * tau * v[i];
        }

        // Check that larfg returns the expected reflection
        CHECK(tlapack::abs(real(w[alphaIdx]) - beta) / tol <
              tlapack::abs(beta));
        CHECK(tlapack::abs(imag(w[alphaIdx])) / tol < one);
        w[alphaIdx] = zero;
        CHECK(tlapack::nrm2(w) / tol < one);
    }
}

TEMPLATE_TEST_CASE("Application of Householder reflectors",
                   "[larf]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);
    using matrix_t = TestType;
    using vector_t = vector_type<TestType>;
    using T = type_t<vector_t>;
    using idx_t = size_type<vector_t>;
    using real_t = real_type<T>;

    // Functors
    Create<vector_t> new_vector;
    Create<matrix_t> new_matrix;

    // Test parameters
    const idx_t m = GENERATE(1, 11, 30);
    const idx_t n = GENERATE(1, 11, 30);
    const Side side = GENERATE(Side::Left, Side::Right);
    const Direction direction =
        GENERATE(Direction::Forward, Direction::Backward);
    const StoreV storeMode = GENERATE(StoreV::Columnwise, StoreV::Rowwise);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " side = " << side
                           << " direction = " << direction
                           << " storeMode = " << storeMode)
    {
        // Constants
        const idx_t k = (side == Side::Left) ? m : n;
        const real_t tol = real_t(4 * std::max(m, n)) * ulp<real_t>();
        const real_t one(1);
        const idx_t oneIdx = (direction == Direction::Forward) ? 0 : k - 1;

        // Vectors
        std::vector<T> v_;
        auto v = new_vector(v_, k);
        std::vector<T> vH_;
        auto vH = new_vector(vH_, k);
        std::vector<T> w_;
        auto w = new_vector(w_, (side == Side::Left) ? n : m);

        // Build v and tau
        for (idx_t i = 0; i < k; ++i)
            v[i] = rand_helper<T>();
        T tau;
        larfg(direction, storeMode, v, tau);
        v[oneIdx] =
            real_t(0xDEADBEEF);  // Put trash in the element that should be one

        // Initialize vH
        for (idx_t i = 0; i < k; ++i)
            vH[i] = conj(v[i]);

        // Initialize matrix C
        std::vector<T> C_;
        auto C = new_matrix(C_, m, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                C(i, j) = rand_helper<T>();

        // Copy C to C0
        std::vector<T> C0_;
        auto C0 = new_matrix(C0_, m, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                C0(i, j) = C(i, j);

        // Apply the Householder reflector
        larf(side, direction, storeMode, v, tau, C);

        // Apply the inverse Householder reflector
        if (side == Side::Left) {
            if (storeMode == StoreV::Columnwise) {
                v[oneIdx] = one;
                gemv(CONJ_TRANS, one, C, v, w);
                ger(-conj(tau), v, w, C);
            }
            else {
                vH[oneIdx] = one;
                gemv(CONJ_TRANS, one, C, vH, w);
                ger(-conj(tau), vH, w, C);
            }
        }
        else {
            if (storeMode == StoreV::Columnwise) {
                v[oneIdx] = one;
                gemv(NO_TRANS, one, C, v, w);
                ger(-conj(tau), w, v, C);
            }
            else {
                vH[oneIdx] = one;
                gemv(NO_TRANS, one, C, vH, w);
                ger(-conj(tau), w, vH, C);
            }
        }

        // Subtract C0 from C
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                C(i, j) -= C0(i, j);

        // Check that larf returns the expected matrix
        CHECK(lange(FROB_NORM, C) / tol < lange(FROB_NORM, C0));
    }
}
