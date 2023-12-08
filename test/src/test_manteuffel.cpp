/// @file test_manteuffel.cpp
/// @author Aslak Djupskås
/// @brief Test Manteuffel eigenvalues 
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/getri.hpp>

// Other routines
#include <tlapack/lapack/gehrd.hpp>
#include <tlapack/lapack/qr_iteration.hpp>


using namespace tlapack;
#include <cmath>

// Define pi
const double pi = std::acos(-1);

TEMPLATE_TEST_CASE("Manteuffel matrix properties",
                   "[manteuffel]", 
                   TLAPACK_TYPES_TO_TEST) 
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using complex_t = complex_type<real_t>;

    typedef real_type<T> real_t;  // equivalent to using real_t = real_type<T>;

    
    
    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    idx_t n = GENERATE(7, 25, 50);
    idx_t m = n * n;
    real_t beta = GENERATE(1, 2, 3, 5, 7, 9);
    real_t h = 1.0;
    real_t L = h*(n+1);


    DYNAMIC_SECTION("n = " << n << " beta = " << beta) 
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(m * m) * eps;

        // Define the matrices and constants
        std::vector<T> M_;
        // Initialize matrix A which is m- by -m
        auto M = new_matrix(M_, m, m);
        
        // Define the matrices and constants
        std::vector<T> N_;
        // Initialize matrix A which is m- by -m
        auto N = new_matrix(N_, m, m);
        
        // Define the matrices and constants
        std::vector<T> A_;
        // Initialize matrix A which is m- by -m
        auto A = new_matrix(A_, m, m);
        

        // Initialize and compute C using the explicit formula
        mm.generateM_manteuffel(M, n);
        mm.generateN_manteuffel(N, n);
        mm.generateManteuffel(A, M, N, m, h, beta);

 
        idx_t i = 0;
        // compute eigenvalues of A
        std::vector<complex_t> evals(m);
        std::complex<double> c(2, 0);

        

        #include <cmath>
        for (idx_t k = 0; k < n; ++k) {
            for (idx_t j = 0; j < n; ++j) {
                std::complex<double> beta_complex = beta; 
                std::complex<double> cos_j = std::cos((j+1) * pi / L);
                std::complex<double> cos_k = std::cos((k+1) * pi / L);
                std::complex<double> argument = 1.0 - std::pow(beta_complex / 2.0, 2.0);
                std::complex<double> term = std::sqrt(argument) * (cos_j + cos_k);
                evals[i] = c * (c - term);
                i++;
            }
        }
        
        // Define tau for Hessenberg reduction
        std::vector<T> tau((m));
        const real_t zero(0);
        const idx_t ilo = 0;
        const idx_t ihi = m;


        // Compute Hessenberg form of A
        int gerr = gehrd(0, m, A, tau);
        CHECK(gerr == 0);

        // Throw away reflectors
        for (idx_t j = 0; j < m; ++j)
            for (idx_t i = j + 2; i < m; ++i)
                A(i, j) = zero;

        // Define Q
        std::vector<T> Q_;
        auto Q = new_matrix(Q_, m, m);
        
        // Define w, which is vector of eigenvalues
        std::vector<complex_t> w(m);

        using variant_t = std::tuple<QRIterationVariant, idx_t, idx_t>;
        const variant_t variant = variant_t(QRIterationVariant::DoubleShift, 0, 0);

        QRIterationOpts opts;
        opts.variant = std::get<0>(variant);

        int ierr = qr_iteration(false, false, ilo, ihi, A, w, Q, opts);
        CHECK(ierr == 0);

        // If the eigenvalues are real, sort them by their real values only
        if (beta == 1 || beta == 2 || beta == 0.5) {
            std::sort(evals.begin(), evals.end(), [](const complex_t& a, const complex_t& b) {
                return a.real() < b.real();
            });
            std::sort(w.begin(), w.end(), [](const complex_t& a, const complex_t& b) {
                return a.real() < b.real();
            });
        
        // If the eigenvalues are complex, sort them by their imaginary parts first, then real parts
        } else {
            std::sort(evals.begin(), evals.end(), [](const complex_t& a, const complex_t& b) {
                return (a.imag() < b.imag()) || (a.imag() == b.imag() && a.real() < b.real());
            });
            std::sort(w.begin(), w.end(), [](const complex_t& a, const complex_t& b) {
                return (a.imag() < b.imag()) || (a.imag() == b.imag() && a.real() < b.real());
            });
        }

        // Compute the residual
        real_t residual = 0.0;
        for (idx_t i = 0; i < m; ++i) {
            residual += std::pow(std::abs(w[i] - evals[i]), 2.0);
        }
        residual = std::sqrt(residual);
        
        // Compute the Frobenius norm 
        real_t norm = 0.0;
        for (idx_t i = 0; i < m; ++i) {
            norm += std::pow(std::abs(evals[i]), 2.0);
        }

        norm = std::sqrt(norm);
        // std::cout << safe_max<real_t>() << std::endl;
        // Print the residual/norm, beta, and safe_max<real_t>(), 
        std::cout <<  "Error = " << residual/norm << "beta =  " << beta << "Precision: " << safe_max<real_t>() << std::endl; 
        // print tol and eps
        std::cout << "tol = " << tol << "eps = " << eps << std::endl;
        // print new line
        std::cout << std::endl;
        // Compute the forward error
        CHECK(residual/norm < tol );
    }
}