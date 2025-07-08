/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
// #include "../../test/include/testutils.hpp"

#include <tlapack/plugins/legacyArray.hpp>
#include "../../test/include/MatrixMarket.hpp"


// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/hemm.hpp>
#include <tlapack/blas/hemm2.hpp>
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lantr.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/lauum_recursive.hpp>
#include <tlapack/lapack/potrf.hpp>
#include <tlapack/lapack/trtri_recursive.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/lapack/trmm_out.hpp>

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
    std::cout << std::endl;
}
//------------------------------------------------------------------------------
// template <TLAPACK_MATRIX matrixA_t,
//           TLAPACK_MATRIX matrixB_t,
//           TLAPACK_MATRIX matrixC_t,
//           TLAPACK_SCALAR alpha_t,
//           TLAPACK_SCALAR beta_t,
//           class T = type_t<matrixB_t>,
//           disable_if_allow_optblas_t<pair<matrixA_t, T>,
//                                      pair<matrixB_t, T>,
//                                      pair<matrixC_t, T>,
//                                      pair<alpha_t, T>,
//                                      pair<beta_t, T> > = 0>
// void trmm_out(Side side,
//               Uplo uplo,
//               Op transA,
//               Diag diag,
//               Op transB,
//               const alpha_t& alpha,
//               const matrixA_t& A,
//               const matrixB_t& B,
//               const beta_t& beta,
//               matrixC_t& C)
// {
//     using idx_t = tlapack::size_type<matrixA_t>;
//     using range = pair<idx_t, idx_t>;
//     using real_t = real_type<T>;

//     idx_t n = nrows(B);
//     idx_t m = ncols(B);
//     idx_t n0 = n / 2;

//     if (n == 1) {
//         // We want to do AXPBY, as of today, this routine is not yet BLAS, and
//         // moreover we need a conjugate on one of the B vector
//         //
//         // this is level 0 of an AXPBY with conjugate option
//         for (idx_t i = 0; i < n; ++i) {
//             for (idx_t j = 0; j < m; ++j) {
//                 C(j, i) = alpha * conj(B(i, j)) * A(0, 0) + beta * C(j, i);
//             }
//         }
//     }
//     else {
//         // std::cout << "C1 rows: " << 0 << ", " << m << " cols: " << n0 << ", "
//         // << n << std::endl;
//         auto C0 = slice(C, range(0, m), range(0, n0));
//         auto C1 = slice(C, range(0, m), range(n0, n));
//         // std::cout << "C1 = " << std::endl;
//         // printMatrix(C1);
//         auto A00 = slice(A, range(0, n0), range(0, n0));
//         auto A10 = slice(A, range(n0, n), range(0, n0));
//         auto A11 = slice(A, range(n0, n), range(n0, n));
//         // std::cout << "A11 = " << std::endl;
//         // printMatrix(A11);
//         auto B0 = slice(B, range(0, n0), range(0, m));
//         auto B1 = slice(B, range(n0, n), range(0, m));
//         // std::cout << "B1 = " << std::endl;
//         // printMatrix(B1);

//         // trmm_out_cheat(Op::ConjTrans, Op::NoTrans, real_t(1), B1, A11,
//         // real_t(1), C1); gemm(Op::ConjTrans, Op::NoTrans, alpha, B1, A11,
//         // beta, C1);
//         trmm_out(side, uplo, transA,  diag, transB, alpha, A11, B1, beta, C1);
//         // std::cout << "C1 after cheat = " << std::endl;
//         // printMatrix(C1);

//         gemm(Op::ConjTrans, Op::NoTrans, alpha, B1, A10, beta, C0);

//         // gemm(Op::ConjTrans, Op::NoTrans, alpha, B0, A00, beta, C0);
//         trmm_out(side, uplo, transA,  diag, transB, alpha, A00, B0, real_t(1),
//                  C0);
//     }
// }

//--------------------------------------------------------------

//---------------------------------------------------------------------------------
// Goal:
template <typename T>
void run(size_t n, size_t m)
{
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using real_t = real_type<T>;

    std::vector<T> B_(n * m);
    tlapack::LegacyMatrix<T> B(n, m, &B_[0], n);

    // std::vector<T> A_(n * n);
    // tlapack::LegacyMatrix<T> A(n, n, &A_[0], n);

    std::vector<T> A_(m * m);
    tlapack::LegacyMatrix<T> A(m, m, &A_[0], m);

    // std::vector<T> C_(m * n);
    // tlapack::LegacyMatrix<T> C(m, n, &C_[0], m);

    // std::vector<T> C_copy_(m * n);
    // tlapack::LegacyMatrix<T> C_copy(m, n, &C_copy_[0], m);

    std::vector<T> C_(n * m);
    tlapack::LegacyMatrix<T> C(n, m, &C_[0], n);

    std::vector<T> C_copy_(n * m);
    tlapack::LegacyMatrix<T> C_copy(n, m, &C_copy_[0], n);

    MatrixMarket mm;

    mm.random(A);
    mm.random(B);
    mm.random(C);

    // for (idx_t j = 0; j < n; j++) {
    //     for (idx_t i = 0; i < j; i++) {
    //         A(i, j) = T(static_cast<float>(0xDEADBEEF));
    //     }
    // }

    std::cout << "A before = " << std::endl;
    printMatrix(A);
    std::cout << "B before = " << std::endl;
    printMatrix(B);
    std::cout << "C before = " << std::endl;
    printMatrix(C);

    lacpy(Uplo::General, C, C_copy);

    real_t normCbefore = lange(Norm::Fro, C_copy);
    real_t normAbefore = lantr(Norm::Fro, Uplo::Upper, Diag::NonUnit, A);
    real_t normBbefore = lange(Norm::Fro, B);

    T beta = -0.393092;
    T alpha = -0.56138;

    
    trmm_out(Side::Right, Uplo::Upper, Op::NoTrans,
             Diag::NonUnit, Op::NoTrans, alpha, A, B, beta, C);
             

    // for (idx_t j = 0; j < n; ++j) {
    //     for (idx_t i = 0; i < j; ++i) {
    //         A(i, j) = T(static_cast<float>(0));
    //     }
    // }

        for (idx_t j = 0; j < m; ++j) {
        for (idx_t i = j + 1; i < m; ++i) {
            A(i, j) = T(static_cast<float>(0));
        }
    }
    
    std::cout << "C_copy = " << std::endl;
    printMatrix(C_copy);
    std::cout << "A = " << std::endl;
    printMatrix(A);
    std::cout << "B = " << std::endl;
    printMatrix(B);

    gemm(Op::NoTrans, Op::NoTrans, alpha, B, A, beta, C_copy);

    std::cout << "my C = " << std::endl;
    printMatrix(C);

    std::cout << "correct C = " << std::endl;
    printMatrix(C_copy);

    // for (idx_t j = 0; j < m; ++j) {
    //     for (idx_t i = 0; i < n; ++i) {
    //         C(i, j) -= C_copy(i, j);
    //     }
    // }

        for (idx_t j = 0; j < m; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            C(i, j) -= C_copy(i, j);
        }
    }

    std::cout << "C subtraction = " << std::endl;
    printMatrix(C);
    real_t normC = lange(Norm::Fro, C);

    std::cout << "norm = "
              << normC / ((real_t(1) * normAbefore * normBbefore) +
                          (real_t(1) * normCbefore))
              << std::endl;
}

//----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int n, m;

    // Default arguments
    n = 3;
    m = 2;

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    // printf("run< float  >( %d, %d )", n, n);
    // run<float>(n, m);
    // printf("-----------------------\n");

    // printf("run< double >( %d, %d )", n, n);
    // run<double>(n, m);
    // printf("-----------------------\n");

    // printf("run< long double >( %d, %d )", n, n);
    // run<long double>(n, m);
    // printf("-----------------------\n");

    printf("run complex< float >( %d, %d )", n, n);
    run<std::complex<float> >(n, m);
    printf("-----------------------\n");

    // printf("run complex< double >( %d, %d )", n, n);
    // run<std::complex<double> >(n, m);
    // printf("-----------------------\n");

    return 0;
}
