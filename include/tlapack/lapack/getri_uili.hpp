/// @file getri_uili.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_getri_uili_HH
#define TLAPACK_getri_uili_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/trtri_recursive.hpp"
#include "tlapack/lapack/ul_mult.hpp"

namespace tlapack {

/** getri_uili calculates the inverse of a general n-by-n matrix A
 *
 *  A is assumed to be in the form L U factors on the input,
 *  trtri is used to invert U and L in place
 *  thereafter, ul_mult is called to calculate U^(-1)L^(-1) in place.
 *
 * @return = 0: successful exit
 * @return = i+1: if U(i,i) is exactly zero.  The triangular
 *          matrix is singular and its inverse can not be computed.
 *
 * @param[in,out] A n-by-n matrix.
 *      On entry, the factors L and U from the factorization A = L U.
 *          L is stored in the lower triangle of A; unit diagonal is not stored.
 *          U is stored in the upper triangle of A.
 *      On exit, inverse of A is overwritten on A.
 *
 * @ingroup computational
 */
template <TLAPACK_MATRIX matrix_t>
int getri_uili(matrix_t& A)
{
    // check arguments
    tlapack_check(nrows(A) == ncols(A));

    // Invert the upper part of A; U
    int info = trtri_recursive(Uplo::Upper, Diag::NonUnit, A);
    if (info != 0) return info;

    // Invert the lower part of A; L which has 1 on the diagonal
    trtri_recursive(Uplo::Lower, Diag::Unit, A);

    // multiply U^{-1} and L^{-1} in place using ul_mult
    ul_mult(A);

    return 0;

}  // getri_uili

}  // namespace tlapack

#endif  // TLAPACK_getri_uili_HH
