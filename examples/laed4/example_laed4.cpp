/// @file example_laed4.cpp
/// @author Brian Dang, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// This must run first
#include <tlapack/plugins/legacyArray.hpp>

// Test utilities and definitions (must come before <T>LAPACK headers)
#include <tlapack/blas/her.hpp>
#include <tlapack/blas/nrm2.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/hetd2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/laed4.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/rscl.hpp>
#include <tlapack/lapack/steqr.hpp>
#include <tlapack/lapack/ungtr.hpp>

//
#include <algorithm>
#include <iomanip>  // for std::setprecision()
#include <iostream>

using namespace tlapack;

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
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

//------------------------------------------------------------------------------
template <typename real_t>
void test_laed4(size_t n)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    // Vectors
    std::vector<real_t> d(n);
    std::vector<real_t> lambda(n);
    std::vector<real_t> e(n - 1);
    std::vector<real_t> u(n);
    std::vector<real_t> work(n);
    std::vector<real_t> tau(n - 1);

    std::vector<real_t> v(n);
    std::vector<real_t> av(n);

    std::vector<real_t> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<real_t> Z_;
    auto Z = new_matrix(Z_, n, n);

    // Create a rho > 0
    real_t rho = 0.5;

    // Create a sorted d
    for (idx_t i = 0; i < n; i++) {
        d[i] = (i + 1) * 2 + i;
    }

    // Create a u of norm 1
    for (idx_t i = 0; i < n; i++) {
        u[i] = 2 * i + 1;
    }
    rscl(nrm2(u), u);

    real_t dlam = 0;
    real_t f = 0;
    std::cout << std::endl;
    for (idx_t i = 0; i < n; i++) {
        laed4(n, i, d, u, work, rho, dlam);
        f = 0;
        for (idx_t j = 0; j < n; j++) {
            f += (u[j] * u[j]) / (d[j] - dlam);
        }
        f *= rho;
        f += 1;
        std::cout << std::setprecision(15);
        std::cout << "Lambda from laed4 " << i << ": " << dlam << " f: " << f
                  << " ";

        // Compute an eigenvector v associated with eigenvalue λ
        for (idx_t j = 0; j < n; j++) {
            v[j] = u[j] / (d[j] - dlam);
        }

        auto nrmv = nrm2(v);

        real_t utv = dot(u, v);

        // Compute || A v - λ v ||₂ / | λ | / || v ||₂
        for (idx_t j = 0; j < n; j++) {
            v[j] = dlam * v[j] - d[j] * v[j] - rho * u[j] * utv;
        }
        std::cout << nrm2(v) / nrmv / abs(dlam) << std::endl;
    }

    printf("-----------------------\n");

    // Create A Matrix = D + rho * u*u^T
    laset(GENERAL, real_t(0.), real_t(0.), A);
    her(LOWER_TRIANGLE, rho, u, A);
    for (idx_t i = 0; i < n; i++) {
        A(i, i) += d[i];
    }

    // Turn A into a Tridiagonal
    hetd2(LOWER_TRIANGLE, A, tau);

    // Extract d from A
    for (idx_t i = 0; i < n; i++) {
        lambda[i] = A(i, i);
    }
    // Extract e from A
    for (idx_t i = 0; i < n - 1; i++) {
        e[i] = A(i + 1, i);
    }

    // find the eigenvalues and eigenvectors of the real symmetric tridiagonal
    // matrix A using steqr
    ungtr(LOWER_TRIANGLE, A, tau);
    steqr(true, lambda, e, A);

    for (idx_t i = 0; i < n; i++) {
        real_t f = 0;
        for (idx_t j = 0; j < n; j++) {
            f += (u[j] * u[j]) / (d[j] - lambda[i]);
        }
        f *= rho;
        f += 1;

        std::cout << "Lambda from steqr " << i << ": " << lambda[i]
                  << " f: " << f << " ";

        auto z = slice(A, range{0, n}, i);
        auto nrmz = nrm2(z);

        real_t utz = dot(u, z);

        // Compute || A v - λ v ||₂ / | λ | / || v ||₂
        for (idx_t j = 0; j < n; j++) {
            z[j] = lambda[i] * z[j] - d[j] * z[j] - rho * u[j] * utz;
        }
        std::cout << nrm2(z) / nrmz / abs(lambda[i]) << std::endl;
    }
}
//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int n;

    // Default arguments
    n = (argc < 2) ? 5 : atoi(argv[1]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("test_laed4< float  >( %d, %d )", n, n);
    test_laed4<float>(n);
    printf("-----------------------\n");

    printf("test_laed4< double >( %d, %d )", n, n);
    test_laed4<double>(n);
    printf("-----------------------\n");

    printf("test_laed4< long double >( %d, %d )", n, n);
    test_laed4<long double>(n);
    printf("-----------------------\n");

    return 0;
}
