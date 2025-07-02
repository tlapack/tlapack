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

TEMPLATE_TEST_CASE("LU factorization of a general m-by-n matrix, blocked",
                   "[ul_mul]",
                   TLAPACK_TYPES_TO_TEST)
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
    idx_t n = GENERATE(2, 5, 30, 50);

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
            d[i] = (i + 1) * 2 + i;
        }
        sort(d.begin(), d.end());

        // Create Random Real u
        srand(3);
        for (idx_t j = 0; j < n; j++) {
            u[j] = rand();
        }
        // Normalize u
        real_t sum = 0;
        for (auto num : u) {
            sum += num * num;
        }
        // u / sqrt(sum)
        for (idx_t i = 0; i < n; i++) {
            u[i] = u[i] / sqrt(sum);
        }

        // Create u*u^T
        for (idx_t j = 0; j < n; j++) {
            A(j, j) += d[j];
            for (idx_t i = 0; i < n; i++) {
                A(i, j) = rho * u[i] * u[j];
            }
        }

        // Turn A into a Tridiagonal
        hetd2(LOWER_TRIANGLE, A, tau);

        // Get the Eigenvalues and Eigenvectors
        real_t dlam = 0;
        real_t f = 0;
        real_t info = 0;
        for (idx_t i = 0; i < n; i++) {
            laed4(n, i, d, u, delta, rho, dlam, info);
            f = 0;
            for (idx_t j = 0; j < n; j++) {
                f += (u[j] * u[j]) / (d[j] - dlam);
            }
            f *= rho;
            f += 1;

            // Compute an eigenvector v associated with eigenvalue λ
            for (idx_t j = 0; j < n; j++) {
                v[j] = u[j] / (d[j] - dlam);
            }

            auto nrmv = nrm2(v);

            real_t utv = 0;
            for (idx_t j = 0; j < n; j++) {
                utv += u[j] * v[j];
            }

            // Compute || A v - λ v ||₂ / | λ | / || v ||₂
            for (idx_t j = 0; j < n; j++) {
                v[j] = dlam * v[j] - d[j] * v[j] - rho * u[j] * utv;
            }
            real_t error = nrm2(v);

            CHECK(error <= tol * nrmv * abs(dlam));
        }
    }
}
