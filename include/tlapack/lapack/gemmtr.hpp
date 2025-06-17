/// @file gemmtr.hpp
/// @author Luis Carlos Gutierrez, Kyle D. Cunningham, and Henricus Bouwmeester
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEMMTR_HH
#define TLAPACK_GEMMTR_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * General triangular matrix-matrix multiply:
 * \[
 *     C := \alpha op(A) \times op(B) + \beta C,
 * \]
 * where $op(X)$ is one of
 *     $op(X) = X$,or
 *     $op(X) = X^H$,
 * alpha and beta are scalars, and A, B, and C are matrices, with
 * $op(A)$ an n-by-k matrix, $op(B)$ a k-by-n matrix, and C an n-by-n matrix.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed to be zero:
 *     - Uplo::Lower: A is lower triangular.
 *     - Uplo::Upper: A is upper triangular.
 *     - Uplo::General is illegal (see gemm() instead).
 *
 * @param[in] transA
 *     The operation $op(A)$ to be used:
 *     - Op::NoTrans:   $op(A) = A$.
 *     - Op::Trans:     $op(A) = A^T$.
 *     - Op::ConjTrans: $op(A) = A^H$.
 *
 * @param[in] transB
 *     The operation $op(B)$ to be used:
 *     - Op::NoTrans:   $op(B) = B$.
 *     - Op::Trans:     $op(B) = B^T$.
 *     - Op::ConjTrans: $op(B) = B^H$.
 *
 * @param[in] alpha Scalar.
 * @param[in] A $op(A)$ is an n-by-k matrix.
 * @param[in] B $op(B)$ is an k-by-n matrix.
 * @param[in] beta Scalar.
 * @param[in,out] C is an n-by-n matrix.
 *
 * @ingroup blas3
 */
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          TLAPACK_SCALAR beta_t,
          class T = type_t<matrixC_t>,
          disable_if_allow_optblas_t<pair<matrixA_t, T>,
                                     pair<matrixB_t, T>,
                                     pair<matrixC_t, T>,
                                     pair<alpha_t, T>,
                                     pair<beta_t, T> > = 0>
void gemmtr(Uplo uplo,
            Op transA,
            Op transB,
            const alpha_t& alpha,
            const matrixA_t& A,
            const matrixB_t& B,
            const beta_t& beta,
            matrixC_t& C)
{
    // data traits
    using TA = type_t<matrixA_t>;
    using TB = type_t<matrixB_t>;
    using idx_t = size_type<matrixA_t>;

    // constants
    const idx_t n = (transB == Op::NoTrans) ? ncols(B) : nrows(B);
    const idx_t k = (transA == Op::NoTrans) ? ncols(A) : nrows(A);

    // check arguments
    tlapack_check_false(uplo != Uplo::Upper && uplo != Uplo::Lower);
    tlapack_check_false(transA != Op::NoTrans && transA != Op::Trans &&
                        transA != Op::ConjTrans);
    tlapack_check_false(transB != Op::NoTrans && transB != Op::Trans &&
                        transB != Op::ConjTrans);
    tlapack_check_false((idx_t)ncols(C) != n && (idx_t)nrows(C) != n);
    tlapack_check_false(
        (idx_t)((transA == Op::NoTrans) ? ncols(A) : nrows(A)) != k);
    tlapack_check_false(
        (idx_t)((transB == Op::NoTrans) ? nrows(B) : ncols(B)) != k);
    tlapack_check_false(
        (idx_t)((transA == Op::NoTrans) ? nrows(A) : ncols(A)) != n);
    tlapack_check_false(
        (idx_t)((transB == Op::NoTrans) ? ncols(B) : nrows(B)) != n);

    // Upper Triangular
    if (uplo == UPPER_TRIANGLE) {
        if (transA == Op::NoTrans) {
            using scalar_t = scalar_type<alpha_t, TB>;
            if (transB == Op::NoTrans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= j; ++i)
                        C(i, j) *= beta;
                    for (idx_t l = 0; l < k; ++l) {
                        const scalar_t alphaTimesblj = alpha * B(l, j);
                        for (idx_t i = 0; i <= j; ++i)
                            C(i, j) += A(i, l) * alphaTimesblj;
                    }
                }
            }
            else if (transB == Op::Trans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= j; ++i)
                        C(i, j) *= beta;
                    for (idx_t l = 0; l < k; ++l) {
                        const scalar_t alphaTimesbjl = alpha * B(j, l);
                        for (idx_t i = 0; i <= j; ++i)
                            C(i, j) += A(i, l) * alphaTimesbjl;
                    }
                }
            }
            else {  // transB == Op::ConjTrans
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= j; ++i)
                        C(i, j) *= beta;
                    for (idx_t l = 0; l < k; ++l) {
                        const scalar_t alphaTimesbjl = alpha * conj(B(j, l));
                        for (idx_t i = 0; i <= j; ++i)
                            C(i, j) += A(i, l) * alphaTimesbjl;
                    }
                }
            }
        }
        else if (transA == Op::Trans) {
            using scalar_t = scalar_type<TA, TB>;

            if (transB == Op::NoTrans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= j; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += A(l, i) * B(l, j);
                        C(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
            }
            else if (transB == Op::Trans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= j; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += A(l, i) * B(j, l);
                        C(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
            }
            else {  // transB == Op::ConjTrans
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= j; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += A(l, i) * conj(B(j, l));
                        C(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
            }
        }
        else {  // transA == Op::ConjTrans

            using scalar_t = scalar_type<TA, TB>;

            if (transB == Op::NoTrans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= j; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += conj(A(l, i)) * B(l, j);
                        C(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
            }
            else if (transB == Op::Trans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= j; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += conj(A(l, i)) * B(j, l);
                        C(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
            }
            else {  // transB == Op::ConjTrans
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= j; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += A(l, i) * B(j, l);
                        C(i, j) = alpha * conj(sum) + beta * C(i, j);
                    }
                }
            }
        }
    }
    else {  // uplo == Uplo::Lower
        if (transA == Op::NoTrans) {
            using scalar_t = scalar_type<alpha_t, TB>;
            if (transB == Op::NoTrans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < n; ++i)
                        C(i, j) *= beta;
                    for (idx_t l = 0; l < k; ++l) {
                        const scalar_t alphaTimesblj = alpha * B(l, j);
                        for (idx_t i = j; i < n; ++i)
                            C(i, j) += A(i, l) * alphaTimesblj;
                    }
                }
            }
            else if (transB == Op::Trans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < n; ++i)
                        C(i, j) *= beta;
                    for (idx_t l = 0; l < k; ++l) {
                        const scalar_t alphaTimesbjl = alpha * B(j, l);
                        for (idx_t i = j; i < n; ++i)
                            C(i, j) += A(i, l) * alphaTimesbjl;
                    }
                }
            }
            else {  // transB == Op::ConjTrans
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < n; ++i)
                        C(i, j) *= beta;
                    for (idx_t l = 0; l < k; ++l) {
                        const scalar_t alphaTimesbjl = alpha * conj(B(j, l));
                        for (idx_t i = j; i < n; ++i)
                            C(i, j) += A(i, l) * alphaTimesbjl;
                    }
                }
            }
        }
        else if (transA == Op::Trans) {
            using scalar_t = scalar_type<TA, TB>;

            if (transB == Op::NoTrans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < n; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += A(l, i) * B(l, j);
                        C(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
            }
            else if (transB == Op::Trans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < n; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += A(l, i) * B(j, l);
                        C(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
            }
            else {  // transB == Op::ConjTrans
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < n; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += A(l, i) * conj(B(j, l));
                        C(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
            }
        }
        else {  // transA == Op::ConjTrans

            using scalar_t = scalar_type<TA, TB>;

            if (transB == Op::NoTrans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < n; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += conj(A(l, i)) * B(l, j);
                        C(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
            }
            else if (transB == Op::Trans) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < n; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += conj(A(l, i)) * B(j, l);
                        C(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
            }
            else {  // transB == Op::ConjTrans
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < n; ++i) {
                        scalar_t sum(0);
                        for (idx_t l = 0; l < k; ++l)
                            sum += A(l, i) * B(j, l);
                        C(i, j) = alpha * conj(sum) + beta * C(i, j);
                    }
                }
            }
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_GEMMTR_HH
