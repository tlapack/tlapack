/// @file lahqz_eig22.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LAHQZ_EIG22_HH__
#define __TLAPACK_LAHQZ_EIG22_HH__

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/lahqr_eig22.hpp"

namespace tlapack {

/** Computes the generalized eigenvalues of a 2x2 pencil (A,B) with B upper
 * triangular
 *
 * Note: in LAPACK, this function is quite complicated, taking a lot of overflow
 * conditions into account. I still need to translate that functionality.
 *
 * @param[in] A 2x2 matrix
 * @param[in] B 2x2 upper triangular matrix
 * @param[out] alpha1 complex number
 * @param[out] alpha2 complex number
 * @param[out] beta1 number
 * @param[out] beta2 number
 *                On exit, (alpha1, beta1), (alpha2, beta2) are the generalized
 *                eigenvalues of the pencil (A,B)
 *
 */
template <TLAPACK_MATRIX A_t, TLAPACK_MATRIX B_t, TLAPACK_SCALAR T>
void lahqz_eig22(const A_t& A,
                 const B_t& B,
                 complex_type<T>& alpha1,
                 complex_type<T>& alpha2,
                 T& beta1,
                 T& beta2)
{
    // Calculate X = AB^{-1}
    auto x00 = A(0, 0) / B(0, 0);
    auto x01 = A(0, 1) / B(1, 1);
    auto x10 = A(1, 0) / B(0, 0);
    auto x11 = A(1, 1) / B(1, 1);
    auto u01 = B(0, 1) / B(1, 1);
    x01 = x01 - u01 * x00;
    x11 = x11 - u01 * x10;

    // Calculate eigenvalues of X
    beta1 = (T)1.;
    beta2 = (T)1.;
    lahqr_eig22(x00, x01, x10, x11, alpha1, alpha2);
}

}  // namespace tlapack

#endif  // __LAHQZ_EIG22_HH__
