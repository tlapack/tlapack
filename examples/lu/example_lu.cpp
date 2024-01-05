/// @file example_lu.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @brief Example using the LU decomposition to compute the inverse of A
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
#include <tlapack/plugins/float8_iee_p.hpp>
#ifdef USE_MPFR
    #include <tlapack/plugins/mpreal.hpp>
#endif

// <T>LAPACK
#include <tlapack/blas/trsm.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include "../../eigen/Eigen/Core"

// C++ headers
#include <iostream>
#include <vector>

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
template <class T, tlapack::Layout L>
void run(size_t n, T scale)
{
    using real_t = tlapack::real_type<T>;
    using idx_t = size_t;
    using range = std::__1::pair<idx_t, idx_t>;

    // Create the n-by-n matrix A
    std::vector<T> A_(n * n);
    tlapack::LegacyMatrix<T, idx_t, L> A(n, n, A_.data(), n);

    //create a matrix for recurrent scaling
    std::vector<float> AR(n,0.0);
    std::vector<float> AS(n,0.0);
    std::vector<float> S_(n*n, 0.0);
    tlapack::LegacyMatrix<float, idx_t, L> S(n, n, S_.data(), n);
    for (size_t i = 0; i < n; i++){
        S(i, i) = 1.0;
    }
    std::vector<float> R_(n*n, 0.0);
    tlapack::LegacyMatrix<float, idx_t, L> R(n, n, R_.data(), n);
    for (size_t i = 0; i < n; i++){
        S(i, i) = 1.0;
    }


    float maxR= 0.0, maxS = 0.0;
    std::vector<float> FG_(n * n);
    tlapack::LegacyMatrix<float, idx_t, L> FG(n, n, FG_.data(), n);

    // std::vector<float> R1_;
    // auto R1 = new_matrix(R1_, m, n);
    // std::vector<float> R2_;
    // auto R2 = new_matrix(R2_, m, n);
    // for(int j = 0; j < n; ++j){
    //     for(int i = 0; i < m; ++i){
    //         R1(i,j) = (static_cast<float>(rand()));
    //         R2(i,j) = (static_cast<float>(rand()));
    //     }
    // }
    //  tlapack::geqr2(R1, tau_buffer);
    // tlapack::ung2r(R1, tau_buffer);
    //  tlapack::geqr2(R2, tau_buffer);
    // tlapack::ung2r(R2, tau_buffer);

    // forming A, a random matrix
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i) {
            FG(i, j) = float(scale)*float(-1 + 2*(rand()%2))*(static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)));            //A(i,j) = A(i,j)*scale;
            //A(i,j) = static_cast<float>(i == j ? 1:0);   --added this as a sanity check
        }
    int count = 0;
    //first we'll perform equilibration
    while(true){
        for(int i = 0; i < n; i++){
            auto b1 = tlapack::rows(FG,range(i,i+1));
            AR[i] = 1/sqrt(tlapack::lange(tlapack::Norm::One, b1));
            auto b2 = tlapack::cols(FG, range(i,i+1));
            AS[i] = 1/sqrt(tlapack::lange(tlapack::Norm::One, b2));
            maxR = AR[i] > maxR ? AR[i] : maxR;
            maxS = AS[i] > maxS ? AS[i] : maxS;

        }
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k++){
                FG(j,k) = FG(j,k)*(AR[j])*(AS[k]);
            }
        }
        for(int i = 0 ; i < n; i++){
            R(i,i) = R(i,i)*AR[i];
            S(i,i) = S(i,i)*AS[i];
        }
        //std::cout << maxR;
        count++;
        if(abs(maxR - 1) < 1 || abs(maxS - 1) < 1 || count > 50) break;
        }

        //next we need to scale by a parameter theta
        float maxA = tlapack::lange(tlapack::Norm::Max, FG);

        
        float normA = tlapack::lange(tlapack::Norm::Inf, FG);


    for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < n; ++i){
            // A(i,j) = static_cast<real_t>(sqrt(float(scale)*0.125)*FG(i,j)/normA);
            A(i,j) = static_cast<real_t>(FG(i,j));
        }
     }
     //printMatrix(A);
   
   
    // Allocate space for the LU decomposition
    std::vector<size_t> piv(n);
    std::vector<size_t> piv_float(n);

    std::vector<T> LU_(n * n);
    tlapack::LegacyMatrix<T, idx_t, L> LU(n, n, LU_.data(), n);

    std::vector<float> LU_float_(n * n);
    tlapack::LegacyMatrix<float, idx_t, L> LU_float(n, n, LU_float_.data(), n);

   
    // Matrix A is kept unchanged
    tlapack::lacpy(tlapack::GENERAL, A, LU);
    tlapack::lacpy(tlapack::GENERAL, FG, LU_float);

     
    int infotoo = tlapack::getrf(LU_float, piv_float);


    if (infotoo != 0) {
        std::cerr << "Matrix could not be factorized in f32 as well!" << std::endl;
        return;
    }
    // Computing the LU decomposition of A
    int info = tlapack::getrf(LU, piv);


    if (info != 0) {
        std::cerr << "Matrix could not be factorized!" << std::endl;
        return;
    }

    
    

    // create X to store invese of A later
    std::vector<T> X_(n * n, T(0));
    tlapack::LegacyMatrix<T, idx_t, L> X(n, n, X_.data(), n);

    
    // step 0: store Identity or Scaling matrix on X
    for (size_t i = 0; i < n; i++){
        X(i, i) = real_t(1);    
        
    }

   

    tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Upper,
                  tlapack::Op::NoTrans, tlapack::Diag::NonUnit, T(1), LU,
                  X);

    tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower,
                  tlapack::Op::NoTrans, tlapack::Diag::Unit, T(1), LU, X);

    
     for (idx_t i = n; i-- > 0;) {
        if (piv[i] != i) {
            auto vect1 = tlapack::row(X, piv[i]);
            auto vect2 = tlapack::row(X, i);
            tlapack::swap(vect1, vect2);
        }
    }

    //FX is meant to store the result of our scaling
    std::vector<float> FX_(n * n, 0.0);
    tlapack::LegacyMatrix<float, idx_t, L> FX(n, n, FX_.data(), n);
    

    

    //create E to store A * X
    std::vector<float> E_(n * n);
    tlapack::LegacyMatrix<float, idx_t, L> E(n, n, E_.data(), n);
    std::vector<float> Ef_(n * n);
    tlapack::LegacyMatrix<float, idx_t, L> Ef(n, n, Ef_.data(), n);
     for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < n; ++i){
            
                E(i,j) = (float(X(i,j))) - FG(i,j);
                Ef(i,j) = float(LU(i,j)) - LU_float(i,j);
        }
            
     }
             //printMatrix(Ef);
           

    //  bool verbose = true;
    //  if (verbose) {
    //     std::cout << std::endl << "A = ";
    //     printMatrix(E);
    //     printMatrix(Ef);
    // }

    // // E <----- A * X - I
    // // tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, real_t(1), A, X,
    // //               E);
    // for (size_t i = 0; i < n; i++)
    //     E(i, i) -= real_t(1);

    // error1 is  || X - A || / ||A||
    float error = tlapack::lange(tlapack::Norm::Inf, E)/normA ;
    float other_error = tlapack::lange(tlapack::Norm::Max, Ef);
    //real_t cond_A = normA* tlapack::lange(tlapack::Norm::Fro, X);
    // Output "
    std::cout << "||A||_F = " << normA << std::endl;
    //std::cout << " k(A) = " << cond_A << std::endl;
    std::cout << "||inv(A)*A - I||_F / ||A||_F = " << error << std::endl;
    std::cout << "float vs 8-bit" << other_error << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    typedef ml_dtypes::float8_e4m3fn float8e4m3fn;
    typedef ml_dtypes::float8_e5m2 float8e5m2;
    int n;
    const tlapack::Layout L = tlapack::Layout::ColMajor;

    // Default arguments
    //n = (argc < 2) ? 100 : atoi(argv[1]);
    n = 500;
   
      // Init random seed
    srand(100);
    std::cout.precision(4);
    std::cout << std::scientific << std::showpos;

    // printf("run< float, L >( %d )\n", n);
    // run<float, L>(n, 1.0);
    // printf("-----------------------\n");

    // printf("run< float, L >( %d )\n", n);
    // run<Eigen::half, L>(n, Eigen::half{1});
    // printf("-----------------------\n");

    //-------------------------------------------
    //print out machine epsilon
    //print out rounding mode
    //get to know semantics of current floats
    //print out norm of A ---done
    //condition number ---done
    //accumulation in different precisions?
    //look at mixed precision for sgemm and trsm
    //------------------------------------------

   
    printf("run< float8e4m3fn, L >( %d )\n", n);
    run<float8e5m2 , L>(n, ml_dtypes::float8_internal::numeric_limits_float8_e5m2::max());
    printf("-----------------------\n");

     printf("run< float8e5m2, L >( %d )\n", n);
    run<float8e5m2 , L>(n, ml_dtypes::float8_internal::numeric_limits_float8_e5m2::max());
    
    printf("-----------------------\n");

    // printf("run< float8e4m3fn, L >( %d )\n", n);
    // run<Eigen::half , L>(n);
    // printf("-----------------------\n");


    //  printf("run<bfloat, L >( %d )\n", n);
    // run<Eigen::half, L>(n, Eigen::half{1});
    // printf("-----------------------\n");

    // printf("run< double, L >( %d )\n", n);
    // run<double, L>(n, 1);
    // printf("-----------------------\n");

    // printf("run< complex<float>, L >( %d )\n", n);
    // run<std::complex<float>, L>(n, 1);
    // printf("-----------------------\n");

    // printf("run< complex<double>, L >( %d )\n", n);
    // run<std::complex<double>, L>(n, 1);
    // printf("-----------------------\n");

// #ifdef USE_MPFR
//     printf("run< mpfr::mpreal, L >( %d )\n", n);
//     run<mpfr::mpreal, L>(n, 1);
//     printf("-----------------------\n");

//     printf("run< complex<mpfr::mpreal>, L >( %d )\n", n);
//     run<std::complex<mpfr::mpreal>, L>(n, mpfr::mpreal(1.0));
//     printf("-----------------------\n");
// #endif
    

    return 0;
}