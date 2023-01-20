/// @file potrs.hpp Apply the Cholesky factorization to solve a linear system.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_POTRS_HH
#define TLAPACK_POTRS_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/trsm.hpp"

namespace tlapack {

/** Apply the Cholesky factorization to solve a linear system.
 * \[
 *      A X = B,
 * \]
 * where
 *      $A = U^H U,$ if uplo = Upper, or
 *      $A = L L^H,$ if uplo = Lower,
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * @tparam uplo_t
 *      Access type: Upper or Lower.
 *      Either Uplo or any class that implements `operator Uplo()`.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A contains the matrix U;
 *      - Uplo::Lower: Lower triangle of A contains the matrix L.
 *      The other triangular part of A is not referenced.
 *
 * @param[in] A
 *      The factor U or L from the Cholesky factorization of A.
 *
 *      - If uplo = Uplo::Upper, the strictly lower
 *      triangular part of A is not referenced.
 *
 *      - If uplo = Uplo::Lower, the strictly upper
 *      triangular part of A is not referenced.
 *
 * @param[in,out] B
 *      On entry, the matrix B.
 *      On exit,  the matrix X.
 *
 * @return = 0: successful exit.
 *
 * @ingroup computational
 */
template <class uplo_t, class matrixA_t, class matrixB_t>
int potrs(uplo_t uplo, const matrixA_t& A, matrixB_t& B)
{
    using T = type_t<matrixB_t>;
    using real_t = real_type<T>;

    // Constants
    const real_t one(1);

    // Check arguments
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(nrows(A) != ncols(A));
    tlapack_check_false(nrows(B) != ncols(A));

    if (uplo == Uplo::Upper) {
        // Solve A*X = B where A = U**H *U.
        trsm(left_side, uplo, conjTranspose, nonUnit_diagonal, one, A, B);
        trsm(left_side, uplo, noTranspose, nonUnit_diagonal, one, A, B);
    }
    else {
        // Solve A*X = B where A = L*L**H.
        trsm(left_side, uplo, noTranspose, nonUnit_diagonal, one, A, B);
        trsm(left_side, uplo, conjTranspose, nonUnit_diagonal, one, A, B);
    }
    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_POTRS_HH