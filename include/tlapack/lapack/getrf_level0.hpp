/// @file getrf_level0.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GETRF_LV0_HH
#define TLAPACK_GETRF_LV0_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** getrf computes an LU factorization of a general m-by-n matrix A
 *  using partial pivoting with row interchanges.
 *
 *  The factorization has the form
 * \[
 *   P A = L U
 * \]
 *  where P is a permutation matrix constructed from our Piv vector, L is lower
 * triangular with unit diagonal elements (lower trapezoidal if m > n), and U is
 * upper triangular (upper trapezoidal if m < n).
 *
 *  This is a Level 0 version of the algorithm.
 *
 * @return  0 if success
 * @return  i+1 if failed to compute the LU on iteration i
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the factors L and U from the factorization A=PLU;
 *      the unit diagonal elements of L are not stored.
 *
 * @param[in,out] Piv is a k-by-1 integer vector where k=min(m,n)
 * and Piv[i]=j where i<=j<=k-1, which means in the i-th iteration of the
 * algorithm, the j-th row needs to be swapped with i
 *
 * @note To construct L and U, one proceeds as in the following steps
 *      1. Set matrices L m-by-k, and U k-by-n be to matrices with all zeros,
 * where k=min(m,n)
 *      2. Set elements on the diagonal of L to 1
 *      3. below the diagonal of A will be copied to L
 *      4. On and above the diagonal of A will be copied to U
 *
 * @ingroup computational
 */
template <class matrix_t, class vector_t>
int getrf_level0(matrix_t& A, vector_t& Piv)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t end = std::min<idx_t>(m, n);

    // check arguments
    tlapack_check((idx_t)size(Piv) >= end);

    // quick return
    if (m <= 0 || n <= 0) return 0;

    for (idx_t j = 0; j < end; j++) {
        // find pivot and swap the row with pivot row
        Piv[j] = j;
        for (idx_t i = j + 1; i < m; i++) {
            if (abs1(A(i, j)) > abs1(A(Piv[j], j))) Piv[j] = i;
        }

        // if nonzero pivot does not exist, return
        if (A(Piv[j], j) == real_t(0)) {
            return j + 1;
        }

        // if the pivot happens to be a Piv[j]>j(Piv[j] not equal to j), then
        // swap j-th row and Piv[j] row of A
        if (Piv[j] != j) {
            for (idx_t i = 0; i < n; i++) {
                T tmp = A(j, i);
                A(j, i) = A(Piv[j], i);
                A(Piv[j], i) = tmp;
            }
        }

        // divide below diagonal part of j-th column by the element on the
        // diagonal(A(j,j))
        for (idx_t i = j + 1; i < m; i++) {
            A(i, j) /= A(j, j);
        }

        // update the submatrix A(j+1:m-1,j+1:n-1)
        for (idx_t row = j + 1; row < m; row++) {
            for (idx_t col = j + 1; col < n; col++) {
                A(row, col) -= A(row, j) * A(j, col);
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GETRF_LV0_HH
