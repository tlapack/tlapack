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
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/hetd2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/laed4.hpp>
#include <tlapack/lapack/steqr.hpp>

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

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    // Vectors
    std::vector<real_t> d(n);
    std::vector<real_t> lambda(n);
    std::vector<real_t> e(n - 1);
    std::vector<real_t> u(n);
    std::vector<real_t> workSpace(n);
    std::vector<real_t> tau(n - 1);

    std::vector<real_t> tempLambda(n);
    std::vector<real_t> v(n);
    std::vector<real_t> av(n);

    std::vector<real_t> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<real_t> D_;
    auto D = new_matrix(D_, n, n);
    std::vector<real_t> E_;
    auto E = new_matrix(E_, n, n);
    std::vector<real_t> Z_;
    auto Z = new_matrix(Z_, n, n);

    // Turn on for Debugging
    bool verbose = true;

    real_t rho = 0.5;

    srand(3);

    // Create Sorted d
    for (idx_t i = 0; i < n; i++) {
        d[i] = (i + 1) * 2 + i;
    }
    if (verbose) {
        std::cout << "\nSorted d = ( ";
        for (auto index : d) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    // Create Random u
    for (idx_t i = 0; i < n; i++) {
        u[i] = 2 * i + 1;
    }
    if (verbose) {
        std::cout << "\nBefore u Norm = ( ";
        for (auto index : u) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
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
    if (verbose) {
        std::cout << "\nAfter u/sqrt(sum) = ( ";
        for (auto index : u) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    // uSum Check
    real_t uSum = 0;
    for (auto num : u) {
        uSum += num * num;
    }
    if (verbose) {
        std::cout << "\nuSum should be 1: " << uSum << std::endl;
    }

    // Create u*u^T
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            A(i, j) = u[i] * u[j];
        }
    }
    if (verbose) {
        std::cout << "\nU*U^T Matrix =";
        printMatrix(A);
        std::cout << std::endl;
    }

    // Create A Matrix = rho * u*u^T
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            A(i, j) = rho * A(i, j);
        }
    }
    // Create A Matrix = D + rho * u*u^T
    for (idx_t i = 0; i < n; i++) {
        A(i, i) += d[i];
    }
    if (verbose) {
        std::cout << "\nA Matrix = D + rho * u*u^T";
        printMatrix(A);
        std::cout << std::endl;
    }

    // Turn A into a Tridiagonal
    hetd2(LOWER_TRIANGLE, A, tau);
    if (verbose) {
        std::cout << "\nA Matrix after hetd2 =";
        printMatrix(A);
        std::cout << std::endl;

        std::cout << "\ntau = ( ";
        for (auto index : tau) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    // Get d from A
    for (idx_t i = 0; i < n; i++) {
        lambda[i] = A(i, i);
    }
    // Get e from A
    for (idx_t i = 0; i < n - 1; i++) {
        e[i] = A(i + 1, i);
    }
    if (verbose) {
        std::cout << "\nd = ( ";
        for (auto index : d) {
            std::cout << index << " ";
        }
        std::cout << ")\n";

        std::cout << "\ne = ( ";
        for (auto index : e) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    real_t dlam = 0;
    real_t f = 0;
    real_t info = 0;
    for (idx_t i = 0; i < n; i++) {
        laed4(n, i, d, u, workSpace, rho, dlam, info);
        f = 0;
        for (idx_t j = 0; j < n; j++) {
            f += (u[j] * u[j]) / (d[j] - dlam);
        }
        f *= rho;
        f += 1;
        std::cout << std::setprecision(15);
        std::cout << "This is Lambda from laed4 " << i << ":" << dlam
                  << " f: " << f << std::endl;

        tempLambda[i] = dlam;

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
        std::cout << nrm2(v) / nrmv / abs(dlam) << std::endl;
    }

    if (verbose) {
        std::cout << "\ntempLambda = ( ";
        for (auto index : tempLambda) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    // find the eigen and eigen vectors of the Tridiagonal A
    // steqr(false, d, e, A);
    steqr(true, lambda, e, Z);
    if (verbose) {
        std::cout << "\nLambda after steqr = ( ";
        for (auto index : lambda) {
            std::cout << index << " ";
        }
        std::cout << ")\n";
    }

    // f(lambda) = 1 + rho[(u_i ^ 2 / (d_1 - lambda)) +
    //                    (u_i ^ n / (d_n - lambda))]
    for (idx_t i = 0; i < n; i++) {
        real_t f = 0;
        for (idx_t j = 0; j < n; j++) {
            f += (u[j] * u[j]) / (d[j] - lambda[i]);
        }
        f *= rho;
        f += 1;

        std::cout << "Lambda from steqr" << i << ": " << lambda[i]
                  << " f: " << f << std::endl;
    }

    // // Create D Matrix
    // for (idx_t j = 0; j < n; j++) {
    //     D(j, j) = d[j];
    // }
    // if (verbose) {
    //     std::cout << "\nD Matrix =";
    //     printMatrix(D);
    //     std::cout << std::endl;
    // }

    // // Create Identity Matrix with lambda
    // for (idx_t j = 0; j < n; j++) {
    //     E(j, j) = 1;
    // }
    // if (verbose) {
    //     std::cout << "\nE Matrix =";
    //     printMatrix(E);
    //     std::cout << std::endl;
    // }

    // // D - E
    // for (idx_t j = 0; j < n; j++) {
    //     for (idx_t i = 0; i < n; i++) {
    //         D(j, i) = D(j, i) - tempLambda[2] * E(j, i);
    //     }
    // }
    // if (verbose) {
    //     std::cout << "\nD Matrix - E Matrix =";
    //     printMatrix(D);
    //     std::cout << std::endl;
    // }

    // // D^-1
    // lacpy(GENERAL, D, LU);

    // // Computing the LU decomposition of LU
    // int info = getrf(LU, piv);
    // if (info != 0) {
    //     std::cerr << "Matrix could not be factorized!" << std::endl;
    //     return;
    // }

    // // solve Ly = u
    // trsm(LEFT_SIDE, LOWER_TRIANGLE, NO_TRANS, UNIT_DIAG, real_t(1), LU, E);
    // trsm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1), LU,
    // E);

    // // X <----- U^{-1}L^{-1}P; swapping columns of X according to piv
    // for (idx_t i = n; i-- > 0;) {
    //     if (piv[i] != i) {
    //         auto vect1 = tlapack::col(E, i);
    //         auto vect2 = tlapack::col(E, piv[i]);
    //         tlapack::swap(vect1, vect2);
    //     }
    // }
    // if (verbose) {
    //     std::cout << "\nE Matrix stores Inverse";
    //     printMatrix(E);
    //     std::cout << std::endl;
    // }
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
