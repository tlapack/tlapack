/// @file potrf2.hpp Computes the Cholesky factorization of a Hermitian positive
/// definite matrix A using the recursive algorithm.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_POTRI_HH
#define TLAPACK_POTRI_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/lauum_recursive.hpp"
#include "tlapack/lapack/potrf2.hpp"
#include "tlapack/lapack/trtri_recursive.hpp"

namespace tlapack {

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A using the recursive algorithm.
 *
 * The factorization has the form
 *     $A = U^H U,$ if uplo = Upper, or
 *     $A = L L^H,$ if uplo = Lower,
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * This is the recursive version of the algorithm. It divides
 * the matrix into four submatrices:
 * \[
 *     A = \begin{bmatrix}
 *             A_{11}  &  A_{12}
 *         \\  A_{21}  &  A_{22}
 *     \end{bmatrix}
 * \]
 * where $A_{11}$ is n1-by-n1 and $A_{22}$ is n2-by-n2,
 * with n1 = n/2 and n2 = n-n1, where n is the order of the matrix A.
 * The subroutine calls itself to factor $A_{11},$
 * updates and scales $A_{21}$ or $A_{12},$
 * updates $A_{22},$
 * and calls itself to factor $A_{22}.$
 *
 * @tparam uplo_t
 *      Access type: Upper or Lower.
 *      Either Uplo or any class that implements `operator Uplo()`.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *      On entry, the Hermitian matrix A.
 *
 *      - If uplo = Uplo::Upper, the strictly lower
 *      triangular part of A is not referenced.
 *
 *      - If uplo = Uplo::Lower, the strictly upper
 *      triangular part of A is not referenced.
 *
 *      - On successful exit, the factor U or L from the Cholesky
 *      factorization $A = U^H U$ or $A = L L^H.$
 *
 * @param[in] opts Options.
 *      Define the behavior of Exception Handling.
 *
 * @return = 0: successful exit
 * @return i, 0 < i <= n, if the leading minor of order i is not
 *     positive definite, and the factorization could not be completed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t>
void potri(Uplo uplo, matrix_t& A)
{
    tlapack_check(uplo == Uplo::Lower || uplo == Uplo::Upper);
    tlapack_check(nrows(A) == ncols(A));

    potrf2(uplo, A);

    trtri_recursive(uplo, Diag::NonUnit, A);

    lauum_recursive(uplo, A);
}

}  // namespace tlapack

#endif  // TLAPACK_POTRI_HH
