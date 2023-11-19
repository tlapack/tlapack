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
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include "../../eigen/Eigen/Core"

// C++ headers
#include <iostream>
#include <vector>


//------------------------------------------------------------------------------
template <class T, tlapack::Layout L>
void run(size_t n)
{
    using real_t = tlapack::real_type<T>;
    using idx_t = size_t;

    // Create the n-by-n matrix A
    std::vector<T> A_(n * n);
    tlapack::LegacyMatrix<T, idx_t, L> A(n, n, A_.data(), n);

    // forming A, a random matrix
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i) {
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            //A(i,j) = static_cast<float>(i == j ? 1:0);   --added this as a sanity check
        }
    real_t normA = tlapack::lange(tlapack::Norm::Fro, A);
    std::cout << "matrix is" << A(0,0) << std::endl;

    // Allocate space for the LU decomposition
    std::vector<size_t> piv(n);
    std::vector<T> LU_(n * n);
    tlapack::LegacyMatrix<T, idx_t, L> LU(n, n, LU_.data(), n);

    // Matrix A is kept unchanged
    tlapack::lacpy(tlapack::GENERAL, A, LU);

    // Computing the LU decomposition of A
    int info = tlapack::getrf(LU, piv);
    if (info != 0) {
        std::cerr << "Matrix could not be factorized!" << std::endl;
        return;
    }

    // create X to store invese of A later
    std::vector<T> X_(n * n, T(0));
    tlapack::LegacyMatrix<T, idx_t, L> X(n, n, X_.data(), n);

    // step 0: store Identity on X
    for (size_t i = 0; i < n; i++)
        X(i, i) = real_t(1);

    // step1: solve L Y = I
    tlapack::trsm(tlapack::Side::Left, tlapack::Uplo::Lower,
                  tlapack::Op::NoTrans, tlapack::Diag::Unit, real_t(1), LU, X);

    // step2: solve U X = Y
    tlapack::trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                  tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(1), LU,
                  X);

    // X <----- U^{-1}L^{-1}P; swapping columns of X according to piv
    for (idx_t i = n; i-- > 0;) {
        if (piv[i] != i) {
            auto vect1 = tlapack::col(X, i);
            auto vect2 = tlapack::col(X, piv[i]);
            tlapack::swap(vect1, vect2);
        }
    }

    // create E to store A * X
    std::vector<T> E_(n * n);
    tlapack::LegacyMatrix<T, idx_t, L> E(n, n, E_.data(), n);

    // E <----- A * X - I
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, real_t(1), A, X,
                  E);
    for (size_t i = 0; i < n; i++)
        E(i, i) -= real_t(1);

    // error1 is  || E || / ||A||
    real_t error = tlapack::lange(tlapack::Norm::Fro, E) / normA;
    real_t cond_A = normA* tlapack::lange(tlapack::Norm::Fro, X);
    // Output "
    std::cout << "||A||_F = " << normA << std::endl;
    std::cout << " k(A) = " << cond_A << std::endl;
    std::cout << "||inv(A)*A - I||_F / ||A||_F = " << error << std::endl;
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
    n = 100;
    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float, L >( %d )\n", n);
    run<float, L>(n);
    printf("-----------------------\n");

    // printf("run< float, L >( %d )\n", n);]
    // run<Eigen::half, L>(n);
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
    run<float8e4m3fn , L>(n);
    printf("-----------------------\n");

    printf("run< float8e4m3fn, L >( %d )\n", n);
    run<Eigen::half , L>(n);
    printf("-----------------------\n");


     printf("run< float8e4m3fn, L >( %d )\n", n);
    run<Eigen::bfloat16 , L>(n);
    printf("-----------------------\n");

    printf("run< double, L >( %d )\n", n);
    run<double, L>(n);
    printf("-----------------------\n");

    printf("run< complex<float>, L >( %d )\n", n);
    run<std::complex<float>, L>(n);
    printf("-----------------------\n");

    printf("run< complex<double>, L >( %d )\n", n);
    run<std::complex<double>, L>(n);
    printf("-----------------------\n");

#ifdef USE_MPFR
    printf("run< mpfr::mpreal, L >( %d )\n", n);
    run<mpfr::mpreal, L>(n);
    printf("-----------------------\n");

    printf("run< complex<mpfr::mpreal>, L >( %d )\n", n);
    run<std::complex<mpfr::mpreal>, L>(n);
    printf("-----------------------\n");
#endif

    return 0;
}
