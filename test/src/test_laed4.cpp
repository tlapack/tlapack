/// @file test_laed4.cpp
/// @author Brian Dang, University of Colorado Denver, USA
/// @brief Test LAED4.
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <algorithm>

// Other routines
#include <tlapack/lapack/hetd2.hpp>
#include <tlapack/lapack/laed4.hpp>

using namespace tlapack;
using namespace std;

template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < n; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

TEMPLATE_TEST_CASE("LAED4", "[stedc,laed4]", TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;  // equivalent to using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    // m and n represent no. rows and columns of the matrices we will be testing
    // respectively
    // idx_t n = GENERATE(2, 5, 30, 50, 100);
    idx_t n = GENERATE(100);

    srand(3);
    real_t rho = real_t(10 * (float)rand() / (float)RAND_MAX);

    DYNAMIC_SECTION("n = " << n << " rho = " << rho)
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(10 * n) * eps;

        // Initialize matrices A, and A_copy to run tests on
        std::vector<real_t> A_;
        auto A = new_matrix(A_, n, n);

        // Vectors
        std::vector<real_t> d(n);
        std::vector<real_t> u(n);
        std::vector<real_t> tau(n - 1);
        std::vector<real_t> delta(n);
        std::vector<real_t> v(n);
        std::vector<real_t> av(n);

        // Create Sorted Random Real d
        for (idx_t i = 0; i < n; i++) {
            d[i] = real_t((i + 1) * 2 + i);
        }
        sort(d.begin(), d.end());

        // Create Random Real u
        srand(3);
        for (idx_t j = 0; j < n; j++) {
            u[j] = real_t(rand() + 1);
        }
        // Normalize u
        real_t sum = real_t(0);
        for (auto num : u) {
            sum += num * num;
        }
        // u / sqrt(sum)
        for (idx_t i = 0; i < n; i++) {
            u[i] = real_t(u[i] / sqrt(sum));
        }

        auto nrmv = nrm2(u);

        // Create u*u^T
        for (idx_t j = 0; j < n; j++) {
            A(j, j) += d[j];
            for (idx_t i = 0; i < n; i++) {
                A(i, j) = real_t(rho * u[i] * u[j]);
            }
        }

        // Turn A into a Tridiagonal
        hetd2(LOWER_TRIANGLE, A, tau);

        // Get the Eigenvalues and Eigenvectors
        real_t dlam = real_t(0);
        real_t f = real_t(0);
        real_t info = real_t(0);
        for (idx_t i = 0; i < n - 1; i++) {
            laed4(n, i, d, u, delta, rho, dlam, info);

            // check #1: check that dlam is a root of the secular equation
            bool too_hard_skip = false;
            f = real_t(0);
            for (idx_t j = 0; j < n; j++) {
                if (d[j] == dlam) too_hard_skip = true;
                f += real_t((u[j] * u[j]) / (d[j] - dlam));
            }
            f *= real_t(rho);
            f += real_t(1);

            // NEED TO SLOVE THIS ISSUE
            //  if (!too_hard_skip) CHECK(abs(f) <= 20 * sqrt(tol));

            // check #2: check that (v,λ) is an eigenpair for A

            // Compute an eigenvector v associated with eigenvalue λ
            // this uses the naive formula and will fail if d[j] is λ, in
            // which case, we skip the check
            too_hard_skip = false;
            for (idx_t j = 0; j < n; j++) {
                if (d[j] == dlam) too_hard_skip = true;
                v[j] = real_t(u[j] / (d[j] - dlam));
            }

            auto nrmv = nrm2(v);

            real_t utv = real_t(0);
            for (idx_t j = 0; j < n; j++) {
                utv += real_t(u[j] * v[j]);
            }

            // Compute || A v - λ v ||₂ / | λ | / || v ||₂
            for (idx_t j = 0; j < n; j++) {
                v[j] = real_t(dlam * v[j] - d[j] * v[j] - rho * u[j] * utv);
            }
            real_t error = real_t(nrm2(v));

            if (!too_hard_skip) CHECK(error <= tol * nrmv * abs(dlam));
        }
    }
}
