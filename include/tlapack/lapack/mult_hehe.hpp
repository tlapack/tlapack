/// @file mult_hehe.hpp
/// @author Ella Addison-Taylor, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MULT_HEHE_HH
#define TLAPACK_MULT_HEHE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/hemm.hpp"
#include "tlapack/blas/hemm2.hpp"

namespace tlapack {

/**
 * Hermitian matrix-Hermitian matrix multiply:
 * \[
 *      C := \alpha A B + \beta C,
 * \]
 * where alpha and beta are scalars, A and B are n-by-n Hermitian matrices and C
 * is an n-by-n matrix
 *
 * @param[in] uplo
 *     What part of the matrix A and B is referenced:
 *     - Uplo::Lower: only the lower triangular part of A and B is referenced.
 *     - Uplo::Upper: only the upper triangular part of A and B is referenced.
 *
 * @param[in] alpha Scalar.
 *
 * @param[in] A n-by-n triangular matrix.
 *
 * @param[in] B n-by-n triangular matrix.
 *
 * @param[in] beta Scalar.
 *
 * @param[in, out] C n-by-n matrix.
 *
 * @ingroup auxiliary
 */

template <TLAPACK_SMATRIX matrixA_t,
          TLAPACK_SMATRIX matrixB_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          TLAPACK_SCALAR beta_t>
void mult_hehe(Uplo uplo,
               const alpha_t& alpha,
               matrixA_t& A,
               matrixB_t& B,
               const beta_t& beta,
               matrixC_t& C)
{
    // using TB = type_t<matrixB_t>;
    using TA = type_t<matrixA_t>;
    using TC = type_t<matrixC_t>;
    typedef tlapack::real_type<TA> real_t;
    using idx_t = tlapack::size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (m != n) return;

    if constexpr (is_complex<TA>) {
        if (n <= 1) {
            C(0, 0) = alpha * real(A(0, 0)) * real(B(0, 0)) + beta * C(0, 0);
            return;
        }

        const idx_t n0 = n / 2;

        if (uplo == Uplo::Upper) {
            const idx_t n0 = n / 2;

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

            // A00*B00 = C00
            mult_hehe(Uplo::Upper, alpha, A00, B00, beta, C00);

            // A01*B01^H + (A00*B00 + C00) = C00
            gemm(Op::NoTrans, Op::ConjTrans, alpha, A01, B01, TC(1), C00);

            // A00*B01 + C01 = C01
            hemm(Side::Left, Uplo::Upper, alpha, A00, B01, beta, C01);

            //(A00*B01 + C01) + A01B11 = C
            hemm(Side::Right, Uplo::Upper, alpha, B11, A01, TC(1), C01);

            // A11 * B01H + C10 = C10
            hemm2(Side::Left, Uplo::Upper, Op::ConjTrans, alpha, A11, B01, beta,
                  C10);  // beta

            // //A01^H * B00 + (A11*B01^H)
            hemm2(Side::Right, Uplo::Upper, Op::ConjTrans, alpha, B00, A01,
                  TC(1), C10);

            // A11*B11
            mult_hehe(Uplo::Upper, alpha, A11, B11, beta, C11);

            // A01^H * B01 + A11*B11
            gemm(Op::ConjTrans, Op::NoTrans, alpha, A01, B01, TC(1), C11);

            return;
        }

        else {
            auto A00 = slice(A, range(0, n0), range(0, n0));
            auto A10 = slice(A, range(n0, n), range(0, n0));
            auto A11 = slice(A, range(n0, n), range(n0, n));

            auto B00 = slice(B, range(0, n0), range(0, n0));
            auto B10 = slice(B, range(n0, n), range(0, n0));
            auto B11 = slice(B, range(n0, n), range(n0, n));

            auto C00 = slice(C, range(0, n0), range(0, n0));
            auto C01 = slice(C, range(0, n0), range(n0, n));
            auto C10 = slice(C, range(n0, n), range(0, n0));
            auto C11 = slice(C, range(n0, n), range(n0, n));

            std::cout << std::endl;

            // A00*B00 = C00
            mult_hehe(Uplo::Lower, alpha, A00, B00, beta, C00);

            // A01^H*B10 + C00 = C00
            gemm(Op::ConjTrans, Op::NoTrans, alpha, A10, B10, TC(1), C00);

            // A10*B00 + C10 = C10
            hemm(Side::Right, Uplo::Lower, alpha, B00, A10, beta, C10);

            // A11*B10 + C10 = C10
            hemm(Side::Left, Uplo::Lower, alpha, A11, B10, TC(1), C10);

            // A00*B01^H + C01 = C01
            hemm2(Side::Left, Uplo::Lower, Op::ConjTrans, alpha, A00, B10, beta,
                  C01);

            // A01^H*B11 + C01 = C01
            hemm2(Side::Right, Uplo::Lower, Op::ConjTrans, alpha, B11, A10,
                  TC(1), C01);

            // A11*B11 = C11
            mult_hehe(Uplo::Lower, alpha, A11, B11, beta, C11);

            // alpha(A10H*B10^H) + 1(C11) = C11
            gemm(Op::NoTrans, Op::ConjTrans, alpha, A10, B10, TC(1), C11);

            return;
        }
    }
    else {
        using TB = type_t<matrixB_t>;
        using TA = type_t<matrixA_t>;
        typedef tlapack::real_type<TA> real_t;
        using idx_t = tlapack::size_type<matrixA_t>;
        using range = pair<idx_t, idx_t>;

        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        if (m != n) return;

        if (n <= 1) {
            C(0, 0) = alpha * real(A(0, 0)) * real(B(0, 0)) + beta * C(0, 0);
            return;
        }

        const idx_t n0 = n / 2;

        if (uplo == Uplo::Upper) {
            const idx_t n0 = n / 2;

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

            // alpha * A00 * B00 + beta * C00 = C00
            mult_hehe(Uplo::Upper, alpha, A00, B00, beta, C00);

            // alpha * A01 * B01^H + C00 = C00
            gemm(Op::NoTrans, Op::ConjTrans, alpha, A01, B01, real_t(1), C00);

            // alpha * A00 * B01 + beta * C01 = C01
            hemm(Side::Left, Uplo::Upper, alpha, A00, B01, beta, C01);

            // alpha * A01 * B11 + C01 = C01
            hemm(Side::Right, Uplo::Upper, alpha, B11, A01, real_t(1), C01);

            // alpha * A11 * B01H + beta * C10 = C10
            hemm2(Side::Left, Uplo::Upper, Op::ConjTrans, alpha, A11, B01, beta,
                  C10);

            // alpha * A01^H * B00 + C10 = C10
            hemm2(Side::Right, Uplo::Upper, Op::ConjTrans, alpha, B00, A01,
                  real_t(1), C10);

            // alpha * A11 * B11 + beta * C11 = C11
            mult_hehe(Uplo::Upper, alpha, A11, B11, beta, C11);

            // A01^H * B01 + C11 = C11
            gemm(Op::ConjTrans, Op::NoTrans, alpha, A01, B01, real_t(1), C11);

            return;
        }

        else {
            auto A00 = slice(A, range(0, n0), range(0, n0));
            auto A10 = slice(A, range(n0, n), range(0, n0));
            auto A11 = slice(A, range(n0, n), range(n0, n));

            auto B00 = slice(B, range(0, n0), range(0, n0));
            auto B10 = slice(B, range(n0, n), range(0, n0));
            auto B11 = slice(B, range(n0, n), range(n0, n));

            auto C00 = slice(C, range(0, n0), range(0, n0));
            auto C01 = slice(C, range(0, n0), range(n0, n));
            auto C10 = slice(C, range(n0, n), range(0, n0));
            auto C11 = slice(C, range(n0, n), range(n0, n));

            std::cout << std::endl;

            // alpha * A00 * B00 + beta * C00 = C00
            mult_hehe(Uplo::Lower, alpha, A00, B00, beta, C00);

            // alpha * A01^H * B10 + C00 = C00
            gemm(Op::ConjTrans, Op::NoTrans, alpha, A10, B10, real_t(1), C00);

            // alpha * A10 * B00 + beta * C10 = C10
            hemm(Side::Right, Uplo::Lower, alpha, B00, A10, beta, C10);

            // alpha * A11 * B10 + C10 = C10
            hemm(Side::Left, Uplo::Lower, alpha, A11, B10, real_t(1), C10);

            // alpha * A00 * B10^H + C01 = C01
            hemm2(Side::Left, Uplo::Lower, Op::ConjTrans, alpha, A00, B10, beta,
                  C01);

            // alpha * A10^H * B11 + beta * C01 = C01
            hemm2(Side::Right, Uplo::Lower, Op::ConjTrans, alpha, B11, A10,
                  real_t(1), C01);

            // alpha * A11 * B11 + beta * C11 = C11
            mult_hehe(Uplo::Lower, alpha, A11, B11, beta, C11);

            // alpha * A10H * B10^H + C11 = C11
            gemm(Op::NoTrans, Op::ConjTrans, alpha, A10, B10, real_t(1), C11);

            return;
        }
    }
}
/**
 *
 * Hermitian matrix-Hermitian matrix multiply:
 * \[
 *      C := \alpha A B,
 * \]
 * where alpha and beta are scalars, A and B are n-by-n Hermitian matrices
 * and C is an n-by-n matrix  .
 *
 * @param[in] uplo Upper or Lower triangular matrix multiplication.
 *
 * @param[in] alpha Scalar.
 *
 * @param[in] A n-by-n triangular matrix.
 *
 * @param[in] B n-by-n triangular matrix.
 *
 * @param[in, out] C n-by-n matrix.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SCALAR alpha_t,
          TLAPACK_SMATRIX matrixA_t,
          TLAPACK_SMATRIX matrixB_t,
          TLAPACK_SMATRIX matrixC_t>
void mult_hehe(
    Uplo uplo, const alpha_t& alpha, matrixA_t& A, matrixB_t& B, matrixC_t& C)
{
    mult_hehe(uplo, alpha, A, B, StrongZero(), C);
}
}  // namespace tlapack
#endif  // TLAPACK_MULT_HEHE_HH
