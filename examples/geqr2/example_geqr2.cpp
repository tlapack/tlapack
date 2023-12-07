/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>


// <T>LAPACK
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

//------------------------------------------------------------------------------
template <typename real_t>
double run(size_t m, size_t n, real_t scale)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Arrays
    std::vector<real_t> tau(n);

    // Matrices
    std::vector<float> FG_;
    auto FG = new_matrix(FG_, m, n);
    std::vector<real_t> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<real_t> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<real_t> Q_;
    auto Q = new_matrix(Q_, m, n);
    std::vector<float> Scal_(n,0.0);
    std::vector<float> sums(n,0.0);

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            A(i, j) = static_cast<float>(0xDEADBEEF);
            Q(i, j) = static_cast<float>(0xCAFED00D);
        }
        for (size_t i = 0; i < n; ++i) {
            R(i, j) = static_cast<float>(0xFEE1DEAD);
        }
        tau[j] = static_cast<float>(0xFFBADD11);
    }

    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < m; ++i){
            // if (i % 2==0)
            // FG(i, j) = 100000*float(-1 + 2*(rand()%2))*(static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)));
            // else 
            FG(i, j) = float(-1 + 2*(rand()%2))*(static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)));
            sums[i] += abs(FG(i,j));
        }
    }

    // Frobenius norm of A
    float normA = tlapack::lange(tlapack::MAX_NORM, FG);
    for(int k = 0; k < n; k++){
        Scal_[k] = sqrt(float(scale)*0.125)/(normA*sums[k]);
        //Scal_[k] = sqrt(float(scale)*0.125)/normA;
    }
    
    std::cout << normA;
     for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < m; ++i){
            A(i,j) = static_cast<real_t>(FG(i,j)*Scal_[j]);
        }
     }
    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(A);
    }

    // Copy A to Q
    tlapack::lacpy(tlapack::GENERAL, A, Q);

    // 1) Compute A = QR (Stored in the matrix Q)

    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        // QR factorization
        tlapack::geqr2(Q, tau);

        // Save the R matrix
        tlapack::lacpy(tlapack::UPPER_TRIANGLE, Q, R);

        // Generates Q = H_1 H_2 ... H_n
        tlapack::ung2r(Q, tau);
        //compute Q in 32 bits
    }
    // Record end time
    auto endQR = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedQR =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endQR - startQR);

    // Compute FLOPS
    double flopsQR =
        (4.0e+00 * ((double)m) * ((double)n) * ((double)n) -
         4.0e+00 / 3.0e+00 * ((double)n) * ((double)n) * ((double)n)) /
        (elapsedQR.count() * 1.0e-9);

    // Print Q and R
    if (verbose) {
        std::cout << std::endl << "Q = ";
        printMatrix(Q);
        std::cout << std::endl << "R = ";
        printMatrix(R);
    }

    double norm_orth_1, norm_repres_1;

    // 2) Compute ||Q'Q - I||_F

    {
        std::vector<real_t> work_;
        auto work = new_matrix(work_, n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        tlapack::laset(tlapack::UPPER_TRIANGLE, 0.0, 1.0, work);
        // work receives Q'Q - I
        tlapack::syrk(tlapack::Uplo::Upper, tlapack::Op::Trans, real_t{1.0}, Q, real_t{-1.0},
                      work);

        // Compute ||Q'Q - I||_F
        norm_orth_1 =
            double(tlapack::lansy(tlapack::MAX_NORM, tlapack::UPPER_TRIANGLE, work));

        if (verbose) {
            std::cout << std::endl << "Q'Q-I = ";
            printMatrix(work);
        }
    }

    // 3) Compute ||QR - A||_F / ||A||_F
     double letscheck;
    {
        std::vector<real_t> work_;
        auto work = new_matrix(work_, m, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // Copy Q to work
        tlapack::lacpy(tlapack::GENERAL, Q, work);

        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper,
                      tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t{1.0}, R,
                      work);
        
        std::vector<float> FE_;
        auto FE = new_matrix(FE_, m, n);
        
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                FE(i, j) = float(work(i,j))/Scal_[j] - FG(i, j);

        norm_repres_1 = double(tlapack::lange(tlapack::MAX_NORM, FE)) / double(normA);
    }

    // *) Output

    std::cout << std::endl;
    std::cout << "time = " << elapsedQR.count() * 1.0e-6 << " ms"
              << ",   GFlop/sec = " << flopsQR * 1.0e-9;
    std::cout << std::endl;
    std::cout << "||QR - A||_F/||A||_F  = " << norm_repres_1
              << ",        ||Q'Q - I||_F  = " << norm_orth_1;
    std::cout << std::endl;
    //std::cout << float(R(10,10)) << std::endl;
 
    

    return norm_repres_1;

    
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    typedef ml_dtypes::float8_e4m3fn float8e4m3fn;
    typedef ml_dtypes::float8_e5m2 float8e5m2;
    int m, n;

    // Default arguments
    m = (argc < 2) ? 300 : atoi(argv[1]);
    n = (argc < 3) ? 300 : atoi(argv[2]);
    double err1 = 0;
    double err2 = 0;
    for (int i = 0; i < 1; i++){
    srand(10);  // Init random seed

    // std::cout.precision(5);
    // std::cout << std::scientific << std::showpos;

     printf("run< float8e4m3fn, L >( %d )\n", n);
     std::cout << "epsilon" << ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::epsilon() << std::endl;
    err1 += run<float8e4m3fn>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max());    
    // printf("-----------------------\n");

     printf("run< float8e5m2, L >( %d )\n", n);
          std::cout << "epsilon" << ml_dtypes::float8_internal::numeric_limits_float8_e5m2::epsilon() << std::endl;

    err2 += run<float8e5m2>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_e5m2::max());    
    }
    //run<Eigen::half>(m,n,Eigen::half{1});
    // printf("-----------------------\n");

    // printf("run< float  >( %d, %d )", m, n);

    // run<float>(m, n, 1.0);
    // printf("-----------------------\n");

    // printf("run< double >( %d, %d )", m, n);
    // run<double>(m, n);
    // printf("-----------------------\n");

    // printf("run< long double >( %d, %d )", m, n);
    // run<long double>(m, n);
    // printf("-----------------------\n");
    std::cout << err1 << std::endl;
    std::cout << err2 << std::endl;
    
    

    return 0;
}
