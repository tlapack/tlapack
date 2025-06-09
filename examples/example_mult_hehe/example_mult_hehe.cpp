/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/lapack/potrf.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/blas/hemm.hpp>
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/trtri_recursive.hpp>
#include <tlapack/lapack/lauum_recursive.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

using namespace tlapack;

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}
//------------------------------------------------------------------------------

template <TLAPACK_SMATRIX matrix_t>
void mult_chc_level0(matrix_t& C) 
{

    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    const idx_t m = nrows(C);
    const idx_t n = ncols(C);

    if (m != n) 
        return;
    else
    {
        for (idx_t j = n; j-- > 0; ) {
            real_t real_part_of_cjj;
            real_part_of_cjj = real(C(j, j)) * real(C(j, j));
            for(idx_t k = 0; k < j; ++k) {
                real_part_of_cjj += real(C(k, j)) * real(C(k, j)) + imag(C(k, j)) * imag(C(k, j));
            }
            C(j,j) = real_part_of_cjj;
            for (idx_t i = j; i-- > 0; ) {
                C(i, j) = conj(C(i, i)) * C(i, j);
                for (idx_t k = i; k-- > 0; ) {
                    C(i, j) += conj(C(k, i)) * C(k, j);
                }
            }
        }
    
    return;
    }
}

//----------------------------------------------------------
template <TLAPACK_SMATRIX matrix_t>
void mult_chc(matrix_t& A)
{
    using T = type_t<matrix_t>;
    typedef tlapack::real_type<T> real_t;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (m != n)
        return;

    if (n <= 1){
        A(0,0) = real(A(0,0))*real(A(0,0));
        return;
    }

    const idx_t n0 = n/2;

    auto A00 = slice(A, range(0, n0), range(0, n0));
    auto A01 = slice(A, range(0, n0), range(n0, n));
    auto A11 = slice(A, range(n0, n), range(n0, n));

    mult_chc(A11);

    herk(Uplo::Upper, Op::ConjTrans, real_t(1),  A01, real_t(1), A11);

    trmm(Side::Left, Uplo::Upper, Op::ConjTrans, Diag::NonUnit, real_t(1), A00, A01);

    mult_chc(A00);

    return;
}

//---------------------------------------------------------------------------
template<TLAPACK_SMATRIX matrixA_t, TLAPACK_SMATRIX matrixB_t, TLAPACK_SMATRIX matrixC_t>
void mult_hehe_cheat(matrixA_t& A, matrixB_t& B, matrixC_t& C)
{
    using TA = type_t<matrixA_t>;
    typedef tlapack::real_type<TA> real_t;
    using idx_t = tlapack::size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;

    gemm(Op::NoTrans, Op::NoTrans, real_t(1), A, B, real_t(0), C);

    return;
}
//----------------------------------------------------------------------------------------
// template<TLAPACK_SMATRIX matrixA_t, TLAPACK_SMATRIX matrixB_t, TLAPACK_SMATRIX matrixC_t>
// void mult_hehe_ella(matrixA_t& A, matrixB_t& B, matrixC_t& C) 
// {
//     mult_hehe()
// }
//---------------------------------------------------------
template<TLAPACK_SMATRIX matrixA_t, TLAPACK_SMATRIX matrixB_t, TLAPACK_SMATRIX matrixC_t, TLAPACK_SCALAR alpha_t, TLAPACK_SCALAR beta_t>
void mult_hehe(const alpha_t& alpha, matrixA_t& A, matrixB_t& B, const beta_t& beta, matrixC_t& C) 
{
    using TB = type_t<matrixB_t>;
    using TA = type_t<matrixA_t>;
    typedef tlapack::real_type<TA> real_t;
    using idx_t = tlapack::size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (m != n)
        return;

    if (n <= 1){
        C(0,0) = alpha * real(A(0,0))*real(B(0,0)) + beta * C(0, 0);
        return;
    }
    const idx_t n0 = n/2;
    
    auto A00 = slice(A, range(0, n0), range(0, n0));
    auto A01 = slice(A, range(0, n0), range(n0, n));
    auto A11 = slice(A, range(n0, n), range(n0, n));

    auto B00 = slice(B, range(0, n0), range(0, n0));
    auto B01 = slice(B, range(0, n0), range(n0, n));
    auto B11 = slice(B, range(n0, n), range(n0, n));
    
    auto C00 = slice(C, range(0, n0), range(0, n0));
    auto C01 = slice(C, range(0, n0), range(n0, n));
    auto C10 = slice(C, range(n0, n), range(0, n0));
    auto C11 = slice(C, range(n0, n), range(n0, n));

    //A00*B00 = C00
    mult_hehe(alpha, A00, B00, beta, C00);

    //A01*B01^H + (A00*B00 + C00) = C00
    gemm(Op::NoTrans, Op::ConjTrans, alpha, A01, B01, real_t(1), C00);

    //A00*B01 + C01 = C01
    hemm(Side::Left, Uplo::Upper, alpha, A00, B01, beta, C01);//beta

    //(A00*B01 + C01) + A01B11 = C
    hemm(Side::Right, Uplo::Upper, alpha, B11, A01, real_t(1), C01);

    //Creating B01^H and A01^H
    std::vector<TB> B01H_((n-n0) * n0);
    tlapack::LegacyMatrix<TB> B01H(n-n0, n0, &B01H_[0], n-n0);

    std::vector<TA> A01H_((n-n0) * n0);
    tlapack::LegacyMatrix<TA> A01H(n-n0, n0, &A01H_[0], n-n0);

    std::cout << std::endl;
    for (idx_t i = 0; i < n-n0; ++i)
        for(idx_t j = 0; j < n0; ++j){
            B01H(i, j) = conj(B01(j, i));
            A01H(i, j) = conj(A01(j, i));
        }

    //A11 * B01H + C10 = C10
    hemm(Side::Left, Uplo::Upper, alpha, A11, B01H, beta, C10); //beta

    // //A01^H * B00 + (A11*B01^H)
    hemm(Side::Right, Uplo::Upper, alpha, B00, A01H, real_t(1), C10);

    // A11*B11
    mult_hehe(alpha, A11, B11, beta, C11);

    //A01^H * B01 + A11*B11
    gemm(Op::ConjTrans, Op::NoTrans, alpha, A01, B01, real_t(1), C11); 

    return;
    
}
//--------------------------------------------------------------
// template<TLAPACK_MATRIX matrixA_t,
//          TLAPACK_MATRIX matrixB_t,
//           TLAPACK_MATRIX matrixC_t,
//           TLAPACK_REAL alpha_t,
//           TLAPACK_REAL beta_t,
//           enable_if_t<(
//                           /* Requires: */
//                           is_real<alpha_t> && is_real<beta_t>),
//                       int> = 0,
//           class T = type_t<matrixC_t>,
//           disable_if_allow_optblas_t<pair<matrixA_t, T>,
//                                      pair<matrixC_t, T>,
//                                      pair<alpha_t, real_type<T> >,
//                                      pair<beta_t, real_type<T> > > = 0>
// // C <- alpha(AB) + beta(C)
// void mult_hehe(Uplo uplo, const alpha_t& alpha, matrixA_t& A, matrixB_t& B, const beta_t& beta, matrixC_t& C) 
// {
//         using TB = type_t<matrixB_t>;
//     using TA = type_t<matrixA_t>;
//     typedef tlapack::real_type<TA> real_t;
//     using idx_t = tlapack::size_type<matrixA_t>;
//     using range = pair<idx_t, idx_t>;

//     const idx_t m = nrows(A);
//     const idx_t n = ncols(A);

//     if (m != n)
//         return;

//     if (n <= 1){
//         C(0,0) = alpha * real(A(0,0))*real(B(0,0));

//         return;
//     }
//     const idx_t n0 = n/2;
    
//     auto A00 = slice(A, range(0, n0), range(0, n0));
//     auto A01 = slice(A, range(0, n0), range(n0, n));
//     auto A11 = slice(A, range(n0, n), range(n0, n));

//     auto B00 = slice(B, range(0, n0), range(0, n0));
//     auto B01 = slice(B, range(0, n0), range(n0, n));
//     auto B11 = slice(B, range(n0, n), range(n0, n));
    
//     auto C00 = slice(C, range(0, n0), range(0, n0));
//     auto C01 = slice(C, range(0, n0), range(n0, n));
//     auto C10 = slice(C, range(n0, n), range(0, n0));
//     auto C11 = slice(C, range(n0, n), range(n0, n));

//     //A00*B00 = C
//     mult_hehe(A00, B00, C00);

//     //A01*B01^H + A00*B00 = C
//     gemm(Op::NoTrans, Op::ConjTrans, real_t(1), A01, B01, real_t(1), C00);

//     //A00*B01 = C
//     hemm(Side::Left, Uplo::Upper, real_t(1), A00, B01, real_t(0), C01);

//     //A00*B01 + A01B11 = C
//     hemm(Side::Right, Uplo::Upper, real_t(1), B11, A01, real_t(1), C01);

//     //Creating B01^H and A01^H
//     std::vector<TB> B01H_((n-n0) * n0);
//     tlapack::LegacyMatrix<TB> B01H(n-n0, n0, &B01H_[0], n-n0);

//     std::vector<TA> A01H_((n-n0) * n0);
//     tlapack::LegacyMatrix<TA> A01H(n-n0, n0, &A01H_[0], n-n0);

//     std::cout << std::endl;
//     for (idx_t i = 0; i < n-n0; ++i)
//         for(idx_t j = 0; j < n0; ++j){
//             B01H(i, j) = conj(B01(j, i));
//             A01H(i, j) = conj(A01(j, i));
//         }

//     //A11 * B01H
//     hemm(Side::Left, Uplo::Upper, real_t(1), A11, B01H, real_t(0), C10);

//     // //A01^H * B00 + A11*B01^H
//     hemm(Side::Right, Uplo::Upper, real_t(1), B00, A01H, real_t(1), C10);

//     // A11*B11
//     mult_hehe(A11, B11, C11);

//     //A01^H * B01 + A11*B11
//     gemm(Op::ConjTrans, Op::NoTrans, real_t(1), A01, B01, real_t(1), C11);

//     for (idx_t i = 0; i < n; ++i) {
//         for (idx_t i = 0; i < n; ++i) {
//         }
//     }

//     return;
// }
//---------------------------------------------------------------------------------
//Goal: 
template <typename T>
void run(size_t n, size_t k)
{
    using std::size_t;
    
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    size_t info;
    typedef tlapack::real_type<T> real_t;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;


    // Turn it off if m or n are large
    //bool verbose = false;
    bool verbose = true;


    std::vector<T> D_(n * n);
    tlapack::LegacyMatrix<T> D(n, n, &D_[0], n);

    std::vector<T> E_(n * n);
    tlapack::LegacyMatrix<T> E(n, n, &E_[0], n);

    std::vector<T> F_(n * n);
    tlapack::LegacyMatrix<T> F(n, n, &F_[0], n);

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
        D(i, j) = i;
        D(j, i) = i;
        E(i, j) = i;
        E(j, i) = i;
        F(i, j) = i + 2;
        F(j, i) = i;
        }
    }
    std::cout << "D =" << std::endl;
    printMatrix(D);
    std::cout << std::endl;
    std::cout << "E =" << std::endl;
    printMatrix(E);
    std::cout << std::endl;
    std::cout << "F before=:" << std::endl;
    printMatrix(F);
    std::cout << std::endl;

    //multiplying two upper triangular hermitian matrices
    //one function with alpha/beta one without so C <- AB and C <- alpha(AB) + beta(C)
    //make it work for upper and lower
    mult_hehe(5, D, E, 10, F);

    std::cout << "F =" << std::endl;
    printMatrix(F);
    std::cout << std::endl;

    // Matrices
    std::vector<T> A_(n * n);
    tlapack::LegacyMatrix<T> A(n, n, &A_[0], n);

    std::vector<T> b_(n * k);
    tlapack::LegacyMatrix<T> b(n, k, &b_[0], n);

    std::vector<T> C_(n * n);
    tlapack::LegacyMatrix<T> C(n, n, &C_[0], n);

    std::vector<T> y_(n * k);
    tlapack::LegacyMatrix<T> y(n, k, &y_[0], n);

    // // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
         for (size_t i = 0; i < n; ++i) {
            if constexpr (tlapack::is_complex<T>)
              A(i, j) = T(static_cast<float>(0xDEADBEEF),
                        static_cast<float>(0xDEADBEEF));
            else
                A(i, j) = T(static_cast<float>(0xDEADBEEF));
     }
    } 

    //Generate a random matrix in A
    for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i <= j; ++i) {
            if constexpr (tlapack::is_complex<T>)
                A(i, j) = T(static_cast<float>(rand())/static_cast<float>(RAND_MAX), static_cast<float>(rand())/static_cast<float>(RAND_MAX));
            else
                A(i, j) = T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        }
        A(j, j) = T(n + static_cast<float>(rand())/static_cast<float>(RAND_MAX));
    }

    // for (size_t j = 0; j < k; ++j)
    //     for (size_t i = 0; i < n; ++i) {
    //         if constexpr (tlapack::is_complex<T>)
    //             b(i, j) = T(static_cast<float>(rand())/static_cast<float>(RAND_MAX), static_cast<float>(rand())/static_cast<float>(RAND_MAX));
    //         else
    //             b(i, j) = T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    //     }

    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(A);
        std::cout << std::endl;
    } 
 }



//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int n, k;


    // Default arguments
    n = (argc < 2) ? 5 : atoi(argv[1]);
    k = (argc < 3) ? 2 : atoi(argv[2]);


    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

     printf("run< float  >( %d, %d )", n, k);
     run<float>(n, k);
     printf("-----------------------\n");

     printf("run< double >( %d, %d )", n, k);
     run<double>(n, k);
     printf("-----------------------\n");

     printf("run< long double >( %d, %d )", n, k);
     run<long double>(n, k);
     printf("-----------------------\n");

    //  printf("run complex< float >( %d, %d )", n, k);
    //  run<std::complex<float> >(n, k);
    //  printf("-----------------------\n");

    //  printf("run complex< double >( %d, %d )", n, k);
    //  run<std::complex<double> >(n, k);
    //  printf("-----------------------\n");

    return 0;
}
