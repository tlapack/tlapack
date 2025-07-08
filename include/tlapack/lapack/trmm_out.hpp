/// @file trmm_out.hpp
/// @author Ella Addison-Taylor, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TRMM_OUT
#define TLAPACK_TRMM_OUT

#include "../../../test/include/MatrixMarket.hpp"
#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"

namespace tlapack {
/**
 *
/**
 * Triangular matrix-matrix multiply:
 * \[
 *     C := \alpha A op(B) + \beta C,
 * \]
 * or
 * \[
 *     C := \alpha op(B) A + \beta C,
 * \]
 * where $op(B)$ is one of
 *     $op(B) = B$,
 *     $op(B) = B^T$, or
 *     $op(B) = B^H$,
 * B is an m-by-n matrix, and A is an m-by-m or n-by-n, unit or non-unit,
 * upper or lower triangular matrix.
 *
 * @param[in] side
 *     Whether $op(A)$ is on the left or right of B:
 *     - Side::Left:  $C = \alpha A op(B) + \beta C$.
 *     - Side::Right: $C = \alpha op(B) A + \beta C$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed to be zero:
 *     - Uplo::Lower: A is lower triangular.
 *     - Uplo::Upper: A is upper triangular.
 *     - Uplo::General is illegal (see gemm() instead).
 * 
 * @param[in] transA
 *     Current functionality only works for:
 *     - Op::NoTrans:   $op(A) = A$.
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 * @param[in] transB
 *     The form of $op(B)$:
 *     - Op::NoTrans:   $op(B) = B$.
 *     - Op::Trans:     $op(B) = B^T$.
 *     - Op::ConjTrans: $op(B) = B^H$.
 *
 * @param[in] alpha Scalar.
 * 
 * @param[in] A
 *     - If side = Left: a m-by-m matrix.
 *     - If side = Right: a n-by-n matrix.
 * 
 * @param[in,out] B A m-by-n matrix.
 * 
 * @param[in] beta Scalar.
 * 
 * @param[in, out] C 
 *     - If transB = NoTrans: a m-by-n matrix.
 *     - If transB = Trans or transB = ConjTrans: a n-by-m matrix.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          TLAPACK_SCALAR beta_t,
          class T = type_t<matrixB_t>,
          disable_if_allow_optblas_t<pair<matrixA_t, T>,
                                     pair<matrixB_t, T>,
                                     pair<matrixC_t, T>,
                                     pair<alpha_t, T>,
                                     pair<beta_t, T> > = 0>
void trmm_out(Side side,
              Uplo uplo,
              Op transA,
              Diag diag,
              Op transB,
              const alpha_t& alpha,
              const matrixA_t& A,
              const matrixB_t& B,
              const beta_t& beta,
              matrixC_t& C)
{
    using idx_t = tlapack::size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;
    using real_t = real_type<T>;
    // only working with transA = notrans right now

    idx_t m, n;
    m = (transB == Op::NoTrans) ? nrows(B) : ncols(B);
    n = (transB == Op::NoTrans) ? ncols(B) : nrows(B);

    if (transB == Op::NoTrans) {
        if (uplo == Uplo::Upper) {
            if (diag == Diag::NonUnit) {
                if (side == Side::Right) {  // notransA, notransB, upper,
                                            // nonunit, right works
                    idx_t n0 = n / 2;
                    if (n == 1) {
                        for (idx_t i = 0; i < m; ++i) {
                                C(i, 0) =
                                    alpha * B(i, 0) * A(0, 0) + beta * C(i, 0);
                        }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A01 = slice(A, range(0, n0), range(n0, n));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, m), range(0, n0));
                        auto B1 = slice(B, range(0, m), range(n0, n));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transB, transA, alpha, B0, A01, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
                else {  // notransA, notransB, upper, nonunit, left works

                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * B(0, j) * A(0, 0) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A01 = slice(A, range(0, m0), range(m0, m));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, m0), range(0, n));
                        auto B1 = slice(B, range(m0, m), range(0, n));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transA, transB, alpha, A01, B1, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
            }
            else {
                if (side == Side::Right) {  // notransA, notransB, upper, unit,
                                            // right works
                    idx_t n0 = n / 2;

                    if (n == 1) {
                        for (idx_t i = 0; i < m; ++i) {
                            C(i, 0) = alpha * B(i, 0) + beta * C(i, 0);
                        }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A01 = slice(A, range(0, n0), range(n0, n));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, m), range(0, n0));
                        auto B1 = slice(B, range(0, m), range(n0, n));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transB, transA, alpha, B0, A01, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
                else {  // notransA, notransB, upper, unit, left works
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * B(0, j) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A01 = slice(A, range(0, m0), range(m0, m));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, m0), range(0, n));
                        auto B1 = slice(B, range(m0, m), range(0, n));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transA, transB, alpha, A01, B1, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
            }
        }
        else {
            if (diag == Diag::NonUnit) {
                if (side == Side::Right) {  // notransA, notransB, right,
                                            // nonunit and lower works

                    idx_t n0 = n / 2;
                    if (n == 1) {
                        for (idx_t i = 0; i < m; ++i) {
                                C(i, 0) =
                                    alpha * B(i, 0) * A(0, 0) + beta * C(i, 0);
                            
                        }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A10 = slice(A, range(n0, n), range(0, n0));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, m), range(0, n0));
                        auto B1 = slice(B, range(0, m), range(n0, n));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transB, transA, alpha, B1, A10, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
                else {  // notransA, notransB, left, nonunit, lower works
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * B(0, j) * A(0, 0) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A10 = slice(A, range(m0, m), range(0, m0));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, m0), range(0, n));
                        auto B1 = slice(B, range(m0, m), range(0, n));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transA, transB, alpha, A10, B0, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }


                }
            }
            else {
                if (side == Side::Right) {  // notransA, notransB, right, unit
                                            // and lower works

                    idx_t n0 = n / 2;
                    if (n == 1) {
                        for (idx_t i = 0; i < m; ++i) {
                                C(i, 0) = alpha * B(i, 0) + beta * C(i, 0);
                            
                        }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A10 = slice(A, range(n0, n), range(0, n0));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, m), range(0, n0));
                        auto B1 = slice(B, range(0, m), range(n0, n));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transB, transA, alpha, B1, A10, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
                else {  // notransA, notransB, left, unit and lower works
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * B(0, j) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A10 = slice(A, range(m0, m), range(0, m0));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, m0), range(0, n));
                        auto B1 = slice(B, range(m0, m), range(0, n));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transA, transB, alpha, A10, B0, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
            }
        }
    }
    else if (transB == Op::ConjTrans) {
        if (uplo == Uplo::Upper) {
            if (diag == Diag::NonUnit) {
                if (side == Side::Right) {  // notransA, conjtransB, nonunit,
                                            // right, upper works

                    idx_t n0 = n / 2;
                    if (n == 1) {
                            for (idx_t j = 0; j < m; ++j) {
                                C(j, 0) = alpha * conj(B(0, j)) * A(0, 0) +
                                          beta * C(j, 0);
                            }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A01 = slice(A, range(0, n0), range(n0, n));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, n0), range(0, m));
                        auto B1 = slice(B, range(n0, n), range(0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transB, transA, alpha, B0, A01, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
                else {  // notransA, conjtransB, nonunit, left, upper works
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * conj(B(j, 0)) * A(0, 0) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A01 = slice(A, range(0, m0), range(m0, m));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, n), range(0, m0));
                        auto B1 = slice(B, range(0, n), range(m0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transA, transB, alpha, A01, B1, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
            }
            else {
                if (side == Side::Right) {  // notransA, conjtransB, unit,
                                            // right, upper works

                    idx_t n0 = n / 2;
                    if (n == 1) {
                            for (idx_t j = 0; j < m; ++j) {
                                C(j, 0) =
                                    alpha * conj(B(0, j)) + beta * C(j, 0);
                            }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A01 = slice(A, range(0, n0), range(n0, n));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, n0), range(0, m));
                        auto B1 = slice(B, range(n0, n), range(0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transB, transA, alpha, B0, A01, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
                else {  // notransA, conjtransB, unit, left, upper works
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * conj(B(j, 0)) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A01 = slice(A, range(0, m0), range(m0, m));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, n), range(0, m0));
                        auto B1 = slice(B, range(0, n), range(m0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transA, transB, alpha, A01, B1, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
            }
        }
        else {
            if (diag == Diag::NonUnit) {
                if (side == Side::Right) {  // notransA, conjtransB, nonunit,
                                            // right, lower works

                    idx_t n0 = n / 2;
                    if (n == 1) {
                            for (idx_t j = 0; j < m; ++j) {
                                C(j, 0) = alpha * conj(B(0, j)) * A(0, 0) +
                                          beta * C(j, 0);
                            }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A10 = slice(A, range(n0, n), range(0, n0));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, n0), range(0, m));
                        auto B1 = slice(B, range(n0, n), range(0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transB, transA, alpha, B1, A10, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
                else {  // notransA, conjtransB, nonunit, left, lower works
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * conj(B(j, 0)) * A(0, 0) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A10 = slice(A, range(m0, m), range(0, m0));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, n), range(0, m0));
                        auto B1 = slice(B, range(0, n), range(m0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transA, transB, alpha, A10, B0, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
            }
            else {
                if (side == Side::Right) {  // notransA, conjtransB, unit,
                                            // right, lower works

                    idx_t n0 = n / 2;
                    if (n == 1) {
                            for (idx_t j = 0; j < m; ++j) {
                                C(j, 0) =
                                    alpha * conj(B(0, j)) + beta * C(j, 0);
                            }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A10 = slice(A, range(n0, n), range(0, n0));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, n0), range(0, m));
                        auto B1 = slice(B, range(n0, n), range(0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transB, transA, alpha, B1, A10, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
                else {  // notransA, conjtransB, unit, left, lower
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * conj(B(j, 0)) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A10 = slice(A, range(m0, m), range(0, m0));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, n), range(0, m0));
                        auto B1 = slice(B, range(0, n), range(m0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transA, transB, alpha, A10, B0, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
            }
        }
    }
    else {
        if (uplo == Uplo::Upper) {
            if (diag == Diag::NonUnit) {
                if (side == Side::Right) {  // notransA, transB, nonunit, right,
                                            // upper works

                    idx_t n0 = n / 2;
                    if (n == 1) {
                            for (idx_t j = 0; j < m; ++j) {
                                C(j, 0) =
                                    alpha * B(0, j) * A(0, 0) + beta * C(j, 0);
                            }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A01 = slice(A, range(0, n0), range(n0, n));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, n0), range(0, m));
                        auto B1 = slice(B, range(n0, n), range(0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transB, transA, alpha, B0, A01, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
                else {  // notransA, transB, nonunit, left, upper
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * B(j, 0) * A(0, 0) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A01 = slice(A, range(0, m0), range(m0, m));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, n), range(0, m0));
                        auto B1 = slice(B, range(0, n), range(m0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transA, transB, alpha, A01, B1, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
            }
            else {
                if (side == Side::Right) {  // notransA, transB, unit, right,
                                            // upper works

                    idx_t n0 = n / 2;
                    if (n == 1) {
                            for (idx_t j = 0; j < m; ++j) {
                                C(j, 0) = alpha * B(0, j) + beta * C(j, 0);
                            }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A01 = slice(A, range(0, n0), range(n0, n));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, n0), range(0, m));
                        auto B1 = slice(B, range(n0, n), range(0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transB, transA, alpha, B0, A01, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
                else {  // notransA, transB, unit, left, upper
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * B(j, 0) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A01 = slice(A, range(0, m0), range(m0, m));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, n), range(0, m0));
                        auto B1 = slice(B, range(0, n), range(m0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transA, transB, alpha, A01, B1, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
            }
        }
        else {
            if (diag == Diag::NonUnit) {
                if (side == Side::Right) {  // notransA, transB, nonunit, right,
                                            // lower works

                    idx_t n0 = n / 2;
                    if (n == 1) {
                            for (idx_t j = 0; j < m; ++j) {
                                C(j, 0) =
                                    alpha * B(0, j) * A(0, 0) + beta * C(j, 0);
                            }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A10 = slice(A, range(n0, n), range(0, n0));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, n0), range(0, m));
                        auto B1 = slice(B, range(n0, n), range(0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transB, transA, alpha, B1, A10, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
                else {  // notransA, transB, nonunit, left, lower
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * B(j, 0) * A(0, 0) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A10 = slice(A, range(m0, m), range(0, m0));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, n), range(0, m0));
                        auto B1 = slice(B, range(0, n), range(m0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transA, transB, alpha, A10, B0, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
            }
            else {
                if (side == Side::Right) {  // notransA, transB, unit, right,
                                            // lower works

                    idx_t n0 = n / 2;
                    if (n == 1) {
                            for (idx_t j = 0; j < m; ++j) {
                                C(j, 0) = alpha * B(0, j) + beta * C(j, 0);
                            }
                    }
                    else {
                        auto C0 = slice(C, range(0, m), range(0, n0));
                        auto C1 = slice(C, range(0, m), range(n0, n));

                        auto A00 = slice(A, range(0, n0), range(0, n0));
                        auto A10 = slice(A, range(n0, n), range(0, n0));
                        auto A11 = slice(A, range(n0, n), range(n0, n));

                        auto B0 = slice(B, range(0, n0), range(0, m));
                        auto B1 = slice(B, range(n0, n), range(0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, beta, C1);

                        gemm(transB, transA, alpha, B1, A10, beta, C0);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, real_t(1), C0);
                    }
                }
                else {  // notransA, transB, unit, left, lower
                    idx_t m0 = m / 2;

                    if (m == 1) {
                        for (idx_t j = 0; j < n; ++j)
                            C(0, j) =
                                alpha * B(j, 0) + beta * C(0, j);
                    }
                    else {
                        auto C0 = slice(C, range(0, m0), range(0, n));
                        auto C1 = slice(C, range(m0, m), range(0, n));

                        auto A00 = slice(A, range(0, m0), range(0, m0));
                        auto A10 = slice(A, range(m0, m), range(0, m0));
                        auto A11 = slice(A, range(m0, m), range(m0, m));

                        auto B0 = slice(B, range(0, n), range(0, m0));
                        auto B1 = slice(B, range(0, n), range(m0, m));

                        trmm_out(side, uplo, transA, diag, transB, alpha, A00,
                                 B0, beta, C0);

                        gemm(transA, transB, alpha, A10, B0, beta, C1);

                        trmm_out(side, uplo, transA, diag, transB, alpha, A11,
                                 B1, real_t(1), C1);
                    }
                }
            }
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_TRMM_OUT