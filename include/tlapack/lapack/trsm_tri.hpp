/// @file trsm_tri.hpp
/// @author Ella Addison-Taylor, Kyle Cunningham, Henricus Bouwmeester
/// University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TRSM_TRI
#define TLAPACK_TRSM_TRI

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/lascl.hpp"
#include "tlapack/lapack/trmm_out.hpp"

namespace tlapack {

/**
 * Solve the triangular matrix-vector equation
 * \[
 *     op(A) X = \alpha B,
 * \]
 * or
 * \[
 *     X op(A) = \alpha B,
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$, or
 *     $op(A) = A^H$,
 * X and B are n-by-n matrices, and A is an n-by-n, unit or non-unit,
 * upper or lower triangular matrix.
 *
 * @param[in] sideA
 *     Whether $op(A)$ is on the left or right of X:
 *     - Side::Left:  $op(A) X = B$.
 *     - Side::Right: $X op(A) = B$.
 *
 * @param[in] uploB
 *     What part of the matrix B is referenced,
 *     the opposite triangle being assumed to be zero:
 *     - UploB::Lower: B is lower triangular.
 *     - UploB::Upper: B is upper triangular.
 *
 * @param[in] transA
 *     The form of $op(A)$:
 *     - Op::NoTrans:   $op(A) = A$.
 *     - Op::Trans:     $op(A) = A^T$.
 *     - Op::ConjTrans: $op(A) = A^H$.
 *
 * @param[in] diagA
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] alpha Scalar.
 *
 * @param[in] A
 *     - A n-by-n matrix.
 *
 * @param[in,out] B
 *      On entry, the n-by-n matrix B.
 *      On exit,  the n-by-n matrix X.
 *
 * @ingroup auxiliary
 */

template <typename matrixA_t, typename matrixB_t, TLAPACK_SCALAR alpha_t>
void trsm_tri(Side sideA,
              Uplo uploB,
              Op transA,
              Diag diagA,
              const alpha_t& alpha,
              const matrixA_t& A,
              matrixB_t& B)
{
    using T = tlapack::type_t<matrixB_t>;
    using idx_t = tlapack::size_type<matrixB_t>;
    using range = std::pair<idx_t, idx_t>;
    using real_t = tlapack::real_type<T>;

    tlapack::Uplo uploA;
    if (transA != tlapack::Op::NoTrans)
        uploA = (uploB == tlapack::Uplo::Upper) ? tlapack::Uplo::Lower
                                                : tlapack::Uplo::Upper;
    else
        uploA = uploB;

    idx_t n = nrows(A);

    // lascl(uploB, T(1.), alpha, B); this does not work

    if (uploB == Uplo::Upper) {
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i = 0; i < j + 1; i++) {
                B(i, j) *= alpha;
            }
        }
    }
    else {
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i = j; i < n; i++) {
                B(i, j) *= alpha;
            }
        }
    }

    if (n == 1) {
        if (diagA == Diag::NonUnit) {
            if (transA == tlapack::Op::ConjTrans)
                B(0, 0) /= conj(A(0, 0));
            else
                B(0, 0) /= A(0, 0);
            return;
        }
        else {
            //     if (transA == tlapack::Op::ConjTrans)
            //     B(0, 0) /= conj(A(0, 0));
            // else
            //     B(0, 0) /= A(0, 0);
            return;
        }
    }

    idx_t nd = n / 2;

    auto A00 = slice(A, range(0, nd), range(0, nd));
    auto A01 = slice(A, range(0, nd), range(nd, n));
    auto A10 = slice(A, range(nd, n), range(0, nd));
    auto A11 = slice(A, range(nd, n), range(nd, n));

    auto B00 = slice(B, range(0, nd), range(0, nd));
    auto B01 = slice(B, range(0, nd), range(nd, n));
    auto B10 = slice(B, range(nd, n), range(0, nd));
    auto B11 = slice(B, range(nd, n), range(nd, n));

    if (sideA == tlapack::Side::Left) {
        if (transA == tlapack::Op::NoTrans) {
            // Form: B := alpha*inv(A)*B
            if (uploB == tlapack::Uplo::Upper) {
                // Left, NoTrans, Upper, Diag
                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A11, B11);

                trmm_out(Side::Right, uploB, Op::NoTrans, Diag::NonUnit, transA,
                         real_t(-1), B11, A01, real_t(1), B01);

                trsm(sideA, uploA, transA, diagA, real_t(1), A00, B01);

                return;
            }
            else {
                // Left, NoTrans, Lower, Diag
                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A11, B11);

                trmm_out(Side::Right, uploB, Op::NoTrans, Diag::NonUnit, transA,
                         real_t(-1), B00, A10, real_t(1), B10);

                trsm(sideA, uploA, transA, diagA, real_t(1), A11, B10);

                return;
            }
        }
        else {
            // Form: B := alpha*inv(A**T)*B
            // Form: B := alpha*inv(A**H)*B
            if (uploB == tlapack::Uplo::Upper) {
                // Left, Trans or ConjTrans, Upper, Diag

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A11, B11);

                trmm_out(Side::Right, uploB, Op::NoTrans, Diag::NonUnit, transA,
                         real_t(-1), B11, A10, real_t(1), B01);

                trsm(sideA, uploA, transA, diagA, real_t(1), A00, B01);

                return;
            }
            else {
                // Left, Trans or ConjTrans, Lower, Diag

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A11, B11);

                trmm_out(Side::Right, uploB, Op::NoTrans, Diag::NonUnit, transA,
                         real_t(-1), B00, A01, real_t(1), B10);

                trsm(sideA, uploA, transA, diagA, real_t(1), A11, B10);
                return;
            }
        }
    }
    else {
        if (transA == tlapack::Op::NoTrans) {
            // Form: B := alpha*B*inv(A)
            if (uploB == tlapack::Uplo::Upper) {
                // Right, NoTrans, Upper, Diag

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A11, B11);

                trmm_out(Side::Left, uploB, Op::NoTrans, Diag::NonUnit, transA,
                         real_t(-1), B00, A01, real_t(1), B01);

                trsm(sideA, uploA, transA, diagA, real_t(1), A11, B01);

                return;
            }
            else {
                // Right, NoTrans, Lower, Diag
                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A11, B11);

                trmm_out(Side::Left, uploB, Op::NoTrans, Diag::NonUnit, transA,
                         real_t(-1), B11, A10, real_t(1), B10);

                trsm(sideA, uploA, transA, diagA, real_t(1), A00, B10);

                return;
            }
        }
        else {
            // Form: B := alpha*B*inv(A**T)
            // Form: B := alpha*B*inv(A**H)

            if (uploB == tlapack::Uplo::Upper) {
                // Right, Trans or ConjTrans, Upper, Diag
                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A11, B11);

                trmm_out(Side::Left, uploB, Op::NoTrans, Diag::NonUnit, transA,
                         real_t(-1), B00, A10, real_t(1), B01);

                trsm(sideA, uploA, transA, diagA, real_t(1), A11, B01);

                return;
            }
            else {
                // Right, Trans or ConjTrans, Lower, Diag

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, real_t(1), A11, B11);

                trmm_out(Side::Left, uploB, Op::NoTrans, Diag::NonUnit, transA,
                         real_t(-1), B11, A01, real_t(1), B10);

                trsm(sideA, uploA, transA, diagA, real_t(1), A00, B10);
                return;
            }
        }
    }
}
}  // namespace tlapack

#endif  // TLAPACK_TRSM_TRI
