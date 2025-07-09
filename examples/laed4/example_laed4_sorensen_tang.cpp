/// @file example_laed4_sorensen_tang.cpp
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
void run(size_t n)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    std::vector<real_t> d1(n);
    std::vector<real_t> d2(n);
    std::vector<real_t> u1(n);
    std::vector<real_t> u2(n);
    std::vector<real_t> laed4Lam(n);
    std::vector<real_t> steqrLam(n);
    std::vector<real_t> e(n - 1);
    std::vector<real_t> work(n);
    std::vector<real_t> tau(n - 1);

    std::vector<real_t> A_;
    auto A = new_matrix(A_, n, n);

    real_t beta = real_t(1.0 / pow(10.0, -3.0));

    // δ = {1, 2 - β, 2 + β, 3 + (1/3)}
    d1[0] = real_t(1);
    d1[1] = real_t(2) - beta;
    d1[2] = real_t(2) + beta;
    d1[3] = real_t(3 + (1.0 / 3.0));
    d2 = d1;

    // w = {2, β, β, 2}
    u1[0] = real_t(2);
    u1[1] = beta;
    u1[2] = beta;
    u1[3] = real_t(2);

    // z = {ζ, ζ, ζ, ζ} = w / ||w||
    real_t sum = real_t(0);
    for (auto num : u1) {
        sum += num * num;
    }
    // u / sqrt(sum)
    for (idx_t i = 0; i < n; i++) {
        u1[i] = u1[i] / sqrt(sum);
    }
    u2 = u1;

    // rho = 1 / ||w||^2
    real_t rho = real_t(1.0) / sum;
    real_t dlam = real_t(0.0);
    for (idx_t i = 0; i < n; i++) {
        laed4(n, i, d1, u1, work, rho, dlam);
        laed4Lam[i] = dlam;
    }

    // Create A Matrix = D + rho * u*u^T
    laset(GENERAL, real_t(0.0), real_t(0.0), A);
    her(LOWER_TRIANGLE, rho, u2, A);
    for (idx_t i = 0; i < n; i++) {
        A(i, i) += d2[i];
    }

    // Turn A into a Tridiagonal
    hetd2(LOWER_TRIANGLE, A, tau);

    // Extract diag from A
    for (idx_t i = 0; i < n; i++) {
        steqrLam[i] = A(i, i);
    }
    // Extract e from A
    for (idx_t i = 0; i < n - 1; i++) {
        e[i] = A(i + 1, i);
    }

    ungtr(LOWER_TRIANGLE, A, tau);
    steqr(false, steqrLam, e, A);

    std::cout << std::setprecision(15);
    std::cout << std::endl;
    for (idx_t i = 0; i < n; i++) {
        std::cout << "dlam = " << laed4Lam[i] << " steqr = " << steqrLam[i]
                  << std::endl;
    }
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int n;

    // Default arguments
    n = (argc < 2) ? 4 : atoi(argv[1]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", n, n);
    run<float>(n);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", n, n);
    run<double>(n);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )", n, n);
    run<long double>(n);
    printf("-----------------------\n");

    return 0;
}
