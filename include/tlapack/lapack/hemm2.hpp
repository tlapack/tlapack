/// @file hemm2.hpp
/// @author Brian Dang, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_HEMM_2_HH
#define TLAPACK_BLAS_HEMM_2_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Hermitian matrix-Hermitian matrix multiply:
 * \[
 *      C := \alpha A B + \beta C,
 * \]
 * \[
 *      C := \alpha B A + \beta C,
 * \]
 * where alpha and beta are scalars, A and B are n-by-n Hermitian matrices and C
 * is an n-by-n matrix
 *
 * @param[in] side
 *      The side the matrix A appears on:
 *     - Side::Left:  $C = \alpha A B + \beta C$,
 *     - Side::Right: $C = \alpha B A + \beta C$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced:
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] transB
 *     The operation $op(B)$ to be used:
 *     - Op::NoTrans:   $op(B) = B$.
 *     - Op::Trans:     $op(B) = B^T$.
 *     - Op::ConjTrans: $op(B) = B^H$.
 *
 * @param[in] alpha Scalar.
 *
 * @param[in] A A n-by-n matrix
 *
 * @param[in] B
 *     - If side = Left and transB = NoTrans or side = Right and transB =
 * Trans/ConjTrans:  B n-by-m matrix.
 *     - If side = Right and transB = NoTrans or side = Left and transB =
 * Trans/ConjTrans: B m-by-n matrix. Imaginary parts of the diagonal
 * elements need not be set, are assumed to be zero on entry, and are set to
 * zero on exit.
 *
 * @param[in] beta Scalar.
 *
 * @param[in, out] C
 *     - If side = Left and transB = NoTrans or side = Right and transB =
 * Trans/ConjTrans:  B n-by-m matrix.
 *     - If side = Right and transB = NoTrans or side = Left and transB =
 * Trans/ConjTrans: B m-by-n matrix.
 *
 * @ingroup blas3
 */

template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          TLAPACK_SCALAR beta_t,
          class T = type_t<matrixC_t>>

void hemm2(Side side,
           Uplo uplo,
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
    using idx_t = size_type<matrixB_t>;

    // constants
    const idx_t m = nrows(B);
    const idx_t n = ncols(B);

    // // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper &&
                        uplo != Uplo::General);
    tlapack_check_false(nrows(A) != ncols(A));
    if ((side == Side::Left && transB == Op::NoTrans) ||
        (side == Side::Right &&
         (transB == Op::Trans || transB == Op::ConjTrans))) {
        tlapack_check_false(ncols(A) != m);
    }
    else {
        tlapack_check_false(nrows(A) != n);
    }
    if (transB == Op::NoTrans) {
        tlapack_check_false(nrows(C) != m);
        tlapack_check_false(ncols(C) != n);
    }
    else {
        tlapack_check_false(nrows(C) != n);
        tlapack_check_false(ncols(C) != m);
    }

    if (side == Side::Left) {
        if (transB == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                // or uplo == Uplo::General
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < m; ++i) {
                        const scalar_type<alpha_t, TB> alphaTimesBij =
                            alpha * B(i, j);
                        scalar_type<TA, TB> sum(0);

                        for (idx_t k = 0; k < i; ++k) {
                            C(k, j) += A(k, i) * alphaTimesBij;
                            sum += conj(A(k, i)) * B(k, j);
                        }
                        C(i, j) = beta * C(i, j) +
                                  real(A(i, i)) * alphaTimesBij + alpha * sum;
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = m - 1; i != idx_t(-1); --i) {
                        const scalar_type<alpha_t, TB> alphaTimesBij =
                            alpha * B(i, j);
                        scalar_type<TA, TB> sum(0);

                        for (idx_t k = i + 1; k < m; ++k) {
                            C(k, j) += A(k, i) * alphaTimesBij;
                            sum += conj(A(k, i)) * B(k, j);
                        }
                        C(i, j) = beta * C(i, j) +
                                  real(A(i, i)) * alphaTimesBij + alpha * sum;
                    }
                }
            }
        }
        else if (transB == Op::Trans) {
            // Trans
            if (uplo == Uplo::Upper) {
                // or uplo == Uplo::General
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < j; i++) {
                            sum += conj(A(i, j)) * B(k, i);
                        }
                        for (idx_t i = j; i < n; i++) {
                            sum += A(j, i) * B(k, i);
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i <= j; i++) {
                            sum += A(j, i) * B(k, i);
                        }
                        for (idx_t i = j + 1; i < n; i++) {
                            sum += conj(A(i, j)) * B(k, i);
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
        }
        else {
            // TransConj
            if (uplo == Uplo::Upper) {
                // or uplo == Uplo::General
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < j; i++) {
                            sum += conj(A(i, j)) * conj(B(k, i));
                        }
                        for (idx_t i = j; i < n; i++) {
                            sum += A(j, i) * conj(B(k, i));
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i <= j; i++) {
                            sum += A(j, i) * conj(B(k, i));
                        }
                        for (idx_t i = j + 1; i < n; i++) {
                            sum += conj(A(i, j)) * conj(B(k, i));
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
        }
    }
    else {  // side == Side::Right
        using scalar_t = scalar_type<alpha_t, TA>;

        if (transB == Op::NoTrans) {
            if (uplo != Uplo::Lower) {
                // uplo == Uplo::Upper or uplo == Uplo::General
                for (idx_t j = 0; j < n; ++j) {
                    {
                        const scalar_t alphaTimesAjj = alpha * real(A(j, j));
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) = beta * C(i, j) + B(i, j) * alphaTimesAjj;
                    }

                    for (idx_t k = 0; k < j; ++k) {
                        const scalar_t alphaTimesAkj = alpha * A(k, j);
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) += B(i, k) * alphaTimesAkj;
                    }

                    for (idx_t k = j + 1; k < n; ++k) {
                        const scalar_t alphaTimesAjk = alpha * conj(A(j, k));
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) += B(i, k) * alphaTimesAjk;
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    {
                        const scalar_t alphaTimesAjj = alpha * real(A(j, j));
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) = beta * C(i, j) + B(i, j) * alphaTimesAjj;
                    }

                    for (idx_t k = 0; k < j; ++k) {
                        const scalar_t alphaTimesAjk = alpha * conj(A(j, k));
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) += B(i, k) * alphaTimesAjk;
                    }

                    for (idx_t k = j + 1; k < n; ++k) {
                        const scalar_t alphaTimesAkj = alpha * A(k, j);
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) += B(i, k) * alphaTimesAkj;
                    }
                }
            }
        }
        else if (transB == Op::Trans) {
            // Trans
            if (uplo == Uplo::Upper) {
                // or uplo == Uplo::General
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < k; i++) {
                            sum += B(i, j) * A(i, k);
                        }
                        for (idx_t i = k; i < m; i++) {
                            sum += B(i, j) * conj(A(k, i));
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < k; i++) {
                            sum += B(i, j) * conj(A(k, i));
                        }
                        for (idx_t i = k; i < m; i++) {
                            sum += B(i, j) * A(i, k);
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
        }
        else {
            // TransConj
            if (uplo == Uplo::Upper) {
                // or uplo == Uplo::General
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < k; i++) {
                            sum += conj(B(i, j)) * A(i, k);
                        }
                        for (idx_t i = k; i < m; i++) {
                            sum += conj(B(i, j)) * conj(A(k, i));
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < k; i++) {
                            sum += conj(B(i, j)) * conj(A(k, i));
                        }
                        for (idx_t i = k; i < m; i++) {
                            sum += conj(B(i, j)) * A(i, k);
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
        }
    }
}

/**
 * Hermitian matrix-Hermitian matrix multiply:
 * \[
 *      C := \alpha A B,
 * \]
 * or
 * \[
 *      C := \alpha B A,
 * \]
 * where alpha is a scalar, A is a n-by-n Hermitian matrix, B and C are n-by-m
 * or m-by-n matrices.
 *
 * @param[in] side
 *      The side the matrix A appears on:
 *     - Side::Left:  $C = \alpha A B$,
 *     - Side::Right: $C = \alpha B A$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced:
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] transB
 *     The operation $op(B)$ to be used:
 *     - Op::NoTrans:   $op(B) = B$.
 *     - Op::Trans:     $op(B) = B^T$.
 *     - Op::ConjTrans: $op(B) = B^H$.
 *
 * @param[in] alpha Scalar.
 *
 * @param[in] A A n-by-n matrix
 *
 * @param[in] B
 *
 *     - If side = Left and transB = NoTrans or side = Right and transB =
 * Trans/ConjTrans:  B n-by-m matrix.
 *     - If side = Right and transB = NoTrans or side = Left and transB =
 * Trans/ConjTrans: B m-by-n matrix. Imaginary parts of the diagonal
 * elements need not be set, are assumed to be zero on entry, and are set to
 * zero on exit.
 *
 * @param[in, out] C
 *     - If side = Left and transB = NoTrans or side = Right and transB =
 * Trans/ConjTrans:  B n-by-m matrix.
 *     - If side = Right and transB = NoTrans or side = Left and transB =
 * Trans/ConjTrans: B m-by-n matrix.
 *
 * @ingroup blas3
 */

template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          class T = type_t<matrixC_t>,
          disable_if_allow_optblas_t<pair<matrixA_t, T>,
                                     pair<matrixB_t, T>,
                                     pair<matrixC_t, T>,
                                     pair<alpha_t, T>>>
void hemm2(Side side,
           Uplo uplo,
           Op transB,
           const alpha_t& alpha,
           const matrixA_t& A,
           const matrixB_t& B,
           matrixC_t& C)
{
    return hemm2(side, uplo, alpha, A, B, StrongZero(), C);
}

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_HEMM_2_HH
