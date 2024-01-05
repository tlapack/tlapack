/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

#include <opencv2/opencv.hpp>


// <T>LAPACK
#include <tlapack/blas/syrk.hpp>
#include <tlapack/lapack/gesvd.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/gemm.hpp>
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
void displayHeatMap(const matrix_t &A, int n) {
    // Assuming m and n are the dimensions of OFef

    // Convert OFef to a grayscale image
    cv::Mat heatMap(n, n, CV_32FC1);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            heatMap.at<float>(i, j) = A(i,j);
        }
    }

    // Normalize the values for visualization purposes
    cv::normalize(heatMap, heatMap, 0, 255, cv::NORM_MINMAX);

    // Display the heat map
    cv::imshow("Heat Map", heatMap);
    cv::waitKey(0);
}
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
double run(size_t m, size_t n, real_t scale, float cond)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Arrays
    std::vector<real_t> tau(n);
    std::vector<float> tau_f(n);
    std::vector<float> tau_buffer(n);

    // Matrices
    std::vector<float> R1_;
    auto R1 = new_matrix(R1_, m, n);
    std::vector<float> R2_;
    auto R2 = new_matrix(R2_, m, n);
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < m; ++i){
            R1(i,j) = (static_cast<float>(rand()));
            R2(i,j) = (static_cast<float>(rand()));
        }
    }
    //take QR of R1 and R2 to get orthogonal matrices U and V^T
    tlapack::geqr2(R1, tau_buffer);
    tlapack::ung2r(R1, tau_buffer);
     tlapack::geqr2(R2, tau_buffer);
    tlapack::ung2r(R2, tau_buffer);

   
    std::vector<float> FG_;
    auto FG = new_matrix(FG_, m, n);
    std::vector<real_t> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<real_t> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<real_t> Q_;
    auto Q = new_matrix(Q_, m, n);
    std::vector<float> Rf_;
    auto Rf = new_matrix(Rf_, n, n);
    std::vector<float> Qf_;
    auto Qf = new_matrix(Qf_, n, n);
    std::vector<float> iA_;
    auto iA = new_matrix(iA_, m, n);

    std::vector<float> Scal_(n,0.0);
    std::vector<float> sums(n,0.0);

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            A(i, j) = static_cast<float>(0xDEADBEEF);
            Q(i, j) = static_cast<float>(0xCAFED00D);
            Qf(i, j) = static_cast<float>(0xCAFED00D);
        }
        for (size_t i = 0; i < n; ++i) {
            R(i, j) = static_cast<float>(0xFEE1DEAD);
            Rf(i, j) = static_cast<float>(0xFEE1DEAD);
        }
        tau[j] = static_cast<float>(0xFFBADD11);
        tau_f[j] = static_cast<float>(0xFFBADD11);
    }

    // Generate a random matrix in A and take it's svd. Then linearly interpolate singular values from desired condition number to 1
    for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i <= j; ++i){
        if(i<=j+1)
         FG(i, j) = (static_cast<float>(rand()));
           
        }
    }
    
    
    std::vector<float> iS_(n*m, 0.0);
    auto iS = new_matrix(iS_, m, n);
    
    //now that we have the SVD, we either scale the rows of Vt or columns of U by the linearly interpolated singular values
    for(int i = 0; i < n; i++){
        iS(i,i) = float(1/(cond - (n - 1- i)*(cond- 1)/(n-1)));
    }
    //printMatrix(iS);
    std::cout << std::endl;
    //now call gemm
    tlapack::gemm(tlapack::NO_TRANS,tlapack::NO_TRANS,1.0, iS, R1,0, iA);
    tlapack::gemm(tlapack::NO_TRANS,tlapack::NO_TRANS,1.0, R2, iA,0, FG);

    for(int i = 0; i < m; i++){
        for( int j = 0; j < n; j++){
            sums[j] += abs(FG(i,j));
        }
    }

    

    
    //once that's done, call gemm and use the new matrix

    

    
    
   
    // Frobenius norm of A
    float normA = tlapack::lange(tlapack::INF_NORM, FG);
    
    for(int k = 0; k < n; k++){
        Scal_[k] = sqrt(float(scale)*0.125)/sums[k];
        //Scal_[k] = sqrt(float(scale)*0.125)/normA;
        //Scal_[k] = 1;
    }
    
    //std::cout << normA;
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
    tlapack::lacpy(tlapack::GENERAL, FG, Qf);

    // 1) Compute A = QR (Stored in the matrix Q)

    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        // QR factorization
        tlapack::geqr2(Q, tau);
         tlapack::geqr2(Qf, tau_f);

        // Save the R matrix
        tlapack::lacpy(tlapack::UPPER_TRIANGLE, Q, R);
        tlapack::lacpy(tlapack::UPPER_TRIANGLE, Qf, Rf);

        // Generates Q = H_1 H_2 ... H_n
        tlapack::ung2r(Q, tau);
        tlapack::ung2r(Qf, tau_f);
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
   

    double norm_orth_1, norm_repres_1;
    double norm_repres_2, norm_repres_3;

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
            double(tlapack::lansy(tlapack::INF_NORM, tlapack::UPPER_TRIANGLE, work));

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
        std::vector<float> FEf_;
        auto FEf = new_matrix(FE_, m, n);
        std::vector<float> oFEf_;
        auto oFEf = new_matrix(oFEf_, m, n);
        for(int i = 0; i < m; i++){
            for(int j = 0; j < i; j++){
                R(i,j)  =0;
                Rf(i,j) = 0;
            }
        }



        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i){
                FE(i, j) = float(work(i,j))/Scal_[j] - FG(i, j);
                FEf(i, j) = Qf(i,j) - float(Q(i,j));
                oFEf(i, j) = Rf(i,j) - float(R(i,j))/Scal_[j];
            }
         if (verbose) {
        std::cout << std::endl << "Q = ";
        printMatrix(Q);
        std::cout << std::endl << "FEf = ";
        printMatrix(FEf);
        std::cout << std::endl << "R = ";
        printMatrix(R);
        std::cout << std::endl << "oFEf = ";
        printMatrix(oFEf);std::cout << std::endl;
    }

        norm_repres_1 = double(tlapack::lange(tlapack::TWO_NORM, FE))/normA ;
        norm_repres_2 = double(tlapack::lange(tlapack::MAX_NORM, FEf));
        norm_repres_3 = double(tlapack::lange(tlapack::MAX_NORM, oFEf));
        //printMatrix(oFEf);
    }

    // *) Output

    // std::cout << std::endl;
    // std::cout << "time = " << elapsedQR.count() * 1.0e-6 << " ms"
    //           << ",   GFlop/sec = " << flopsQR * 1.0e-9;
    // std::cout << std::endl;
    // std::cout << "||QR - A||_F/||A||_F  = " << norm_repres_1
    //           << ",        ||Q'Q - I||_F  = " << norm_orth_1 << std::endl;
    // std::cout << "error in Q wrt float version " <<  norm_repres_2 << std::endl;
    // std::cout << "error in R wrt float version " <<  norm_repres_3 << std::endl;
    // std::cout << std::endl;
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
    m = (argc < 2) ? 10 : atoi(argv[1]);
    n = (argc < 3) ?  10 : atoi(argv[2]);
    double err1 = 0;
    double er3 = 0;
    double err2 = 0;
    for (int i = 0; i < 1; i++){
    srand(100);  // Init random seed

    // std::cout.precision(5);
    // std::cout << std::scientific << std::showpos;

     //printf("run< float8e4m3fn, L >( %d )\n", n);
     std::cout << "epsilon" << ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::epsilon() << std::endl;
    err1 += run<float8e4m3fn>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max(), 1000000000.0);    
    // printf("-----------------------\n");

     //printf("run< float8e5m2, L >( %d )\n", n);
    std::cout << "epsilon" << ml_dtypes::float8_internal::numeric_limits_float8_e5m2::epsilon() << std::endl;
    
    err2 += run<float8e5m2>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_e5m2::max(), 100000000000.0);  

    //er3 +=   run<float>(m,n,1.0);
    }
    //run<Eigen::half>(m,n,Eigen::half{1});
    // printf("-----------------------\n");

    // printf("run< float  >( %d, %d )", m, n);

    er3 += run<float>(m, n, 1.0, 10000.0);
    printf("-----------------------\n");

    // printf("run< double >( %d, %d )", m, n);
    // run<double>(m, n);
    // printf("-----------------------\n");

    // printf("run< long double >( %d, %d )", m, n);
    // run<long double>(m, n);
    // printf("-----------------------\n");
    
    std::cout << err1 << std::endl;
    std::cout << err2 << std::endl;
    std::cout << er3 << std::endl;
    //std::cout << er3 << std::endl;
    //std::cout << float(ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::infinity()) <<std::endl;
    
    

    return 0;
}
