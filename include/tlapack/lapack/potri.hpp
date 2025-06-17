/// @file potrfi.hpp Computes the Inverse of a Hermitian positive
/// definite matrix A using recursive algorithms.
/// @author Eleanor Addison-Taylor, University of Colorado Denver, USA
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

/** Computes the Inverse of a Hermitian
 * positive definite matrix A using recursive algorithms.
 *
 * The inverse has the form
 *     $U^-1,$ if uplo = Upper, or
 *     $L^-1,$ if uplo = Lower,
 * where U is an upper triangular matrix and L is lower triangular.
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
 *      - On successful exit, the inverse of U or L.
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
