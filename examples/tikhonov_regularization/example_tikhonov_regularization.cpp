/// @file example_tikhonov_regularization.cpp
/// @author L. Carlos Gutierrez, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tlapack/plugins/legacyArray.hpp>

#include "../../test/include/MatrixMarket.hpp"

// Check function created for the example
#include "tik_check.hpp"

// Least square solver function created for the example
// #include <tlapack/lapack/tik_bidiag_elden.hpp>
// #include <tlapack/lapack/tik_qr.hpp>
// #include <tlapack/lapack/tik_svd.hpp>
#include <tlapack/lapack/tkhnv.hpp>

#include "tik_chol.hpp"

using namespace tlapack;

//------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t k)
{
    using real_t = real_type<T>;
    using matrix_t = LegacyMatrix<T>;
    using idx_t = size_type<matrix_t>;

    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    // Declare Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> b_;
    auto b = new_matrix(b_, m, k);
    std::vector<T> bcopy_;
    auto bcopy = new_matrix(bcopy_, m, k);
    std::vector<T> x_;
    auto x = new_matrix(x_, n, k);

    // Initializing matrices randomly
    MatrixMarket mm;
    mm.random(A);
    mm.random(b);

    // Create copies for check
    lacpy(GENERAL, A, A_copy);
    lacpy(GENERAL, b, bcopy);

    // Initialize scalars
    real_t lambda(4);

    // Choose method to solve least squares problem of Tikhonov regularized
    // matrix
    std::string method;

    int option = 3;

    switch (option) {
        case 1:
            method = "Tikhonov QR";
            break;
        case 2:
            method = "Tikhonov Bidiag Eldén";
            break;
        case 3:
            method = "Tikhonov SVD";
            break;
        case 4:
            method = "Tikhonov Cholesky";
            break;
        default:
            method = "No method chosen";
    }

    // Outputs method used to solve Least Squares
    std::cout << "\n\nSolving Least Squares using method: " << method << "\n";

    // Executes desired subroutine
    if (method == "Tikhonov QR") {
        tkhnv(A, b, lambda, TikOpts(TikVariant::QR));
    }
    else if (method == "Tikhonov Bidiag Eldén") {
        tkhnv(A, b, lambda, TikOpts(TikVariant::Elden));
    }
    else if (method == "Tikhonov SVD") {
        tkhnv(A, b, lambda, TikOpts(TikVariant::SVD));
    }
    else if (method == "Tikhonov Cholesky") {
        tik_chol(A, b, lambda, x);
        // Solution for check is stored in first n rows of b
        lacpy(GENERAL, x, b);
    }
    // Conducts check for least squares problem
    lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), x);
    tik_check(A_copy, bcopy, lambda, x);
}
//------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n, k;

    // Default arguments
    m = 5;
    n = 3;
    k = 2;

    // Init random seed
    srand(3);

    // Set output format
    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    // Execute run for different variable types
    printf("----------------------------------------------------------\n");
    printf("run< float  >( %d, %d, %d )", m, n, k);
    run<float>(m, n, k);
    printf("----------------------------------------------------------\n");

    printf("run< double >( %d, %d, %d )", m, n, k);
    run<double>(m, n, k);
    printf("----------------------------------------------------------\n");

    printf("run< long double >( %d, %d, %d )", m, n, k);
    run<long double>(m, n, k);
    printf("----------------------------------------------------------\n");

    printf("run< complex<float> >( %d, %d, %d )", m, n, k);
    run<std::complex<float>>(m, n, k);
    printf("----------------------------------------------------------\n");

    printf("run< complex<double> >( %d, %d, %d )", m, n, k);
    run<std::complex<double>>(m, n, k);
    printf("----------------------------------------------------------\n");
    return 0;
}
