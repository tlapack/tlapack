/// @file getrf_recursive.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GETRF_RECURSIVE_HH
#define TLAPACK_GETRF_RECURSIVE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/rscl.hpp"

namespace tlapack {

/** getrf_recursive computes an LU factorization of a general m-by-n matrix A
 *  using partial pivoting with row interchanges.
 *
 *  The factorization has the form
 * \[
 *   P A = L U
 * \]
 *  where P is a permutation matrix constructed from our piv vector, L is lower
 * triangular with unit diagonal elements (lower trapezoidal if m > n), and U is
 * upper triangular (upper trapezoidal if m < n).
 *
 *  This is a recursive version of the algorithm.
 *
 * @return  0 if success
 * @return  i+1 if failed to compute the LU on iteration i
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the factors L and U from the factorization A=PLU;
 *      the unit diagonal elements of L are not stored.
 *
 * @param[in,out] piv is a k-by-1 integer vector where k=min(m,n)
 * and piv[i]=j where i<=j<=k-1, which means in the i-th iteration of the
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
template <class matrix_t, class piv_t>
int getrf_recursive(matrix_t& A, piv_t& piv)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // Using the following lines to pass the abs function to iamax
    // TODO: Replace the following lines by a lambda function if we adopt C++17
    struct abs_f {
        inline constexpr real_t operator()(const T& x) const
        {
            return tlapack::abs(x);
        }
    };
    abs_f absf;
    iamax_opts_t<abs_f> optsIamax(absf);

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = std::min<idx_t>(m, n);

    // check arguments
    tlapack_check((idx_t)size(piv) >= k);

    // quick return
    if (m <= 0 || n <= 0) return 0;

    // base case of recursion; one column matrices or one row matrices
    // one-row matrices
    if (m == 1) {
        // piv has one element
        piv[0] = 0;
        if (A(piv[0], 0) == real_t(0)) {
            // in case which A(0,0) is zero, then we return 1 since in the first
            // iteration we stopped
            return 1;
        }

        return 0;
    }

    // one-column matrices
    else if (n == 1) {
        // when n==1, piv has one element, piv[0] needs to be swapped by the
        // first row
        piv[0] = iamax(col(A, 0), optsIamax);

        // in the following case all elements are zero, and we return 1
        if (A(piv[0], 0) == real_t(0)) return 1;

        // in this case, we can safely swap since A(piv[0],0) is not zero
        if (piv[0] != 0) std::swap(A(piv[0], 0), A(0, 0));

        // by the previous comment, we can safely scale all elements of 0th
        // column by 1/A(0,0)
        auto l = slice(A, range<idx_t>(1, m), 0);
        rscl(A(0, 0), l);

        return 0;
    }

    // the case where m<n, we simply slice A into two parts, A0, a square matrix
    // and A1 where A=[A0 , A1]
    else if (m < n) {
        auto A0 = tlapack::cols(A, tlapack::range<idx_t>(0, m));
        auto A1 = tlapack::cols(A, tlapack::range<idx_t>(m, n));

        int info = getrf_recursive(A0, piv);
        if (info != 0) return info;

        // swap the rows of A1 according to piv
        for (idx_t j = 0; j < k; j++) {
            if (piv[j] != j) {
                auto vect1 = tlapack::row(A1, j);
                auto vect2 = tlapack::row(A1, piv[j]);
                tlapack::swap(vect1, vect2);
            }
        }

        // Solve triangular system A0 X = A1 and update A1
        trsm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::Unit, T(1), A0, A1);

        return 0;
    }
    else {
        // Dimensions for the submatrices
        idx_t k0 = k / 2;

        // in this step, we break A into two matrices, A=[A0 , A1]
        auto A0 = tlapack::cols(A, tlapack::range<idx_t>(0, k0));
        auto A1 = tlapack::cols(A, tlapack::range<idx_t>(k0, n));

        // piv0 is the first k0 elements of piv
        auto piv0 = tlapack::slice(piv, tlapack::range<idx_t>(0, k0));

        // Apply getrf on the left of half of the matrix
        int info = getrf_recursive(A0, piv0);
        if (info != 0) return info;

        // swap the rows of A1
        for (idx_t j = 0; j < k0; j++) {
            if (piv0[j] != j) {
                auto vect1 = tlapack::row(A1, j);
                auto vect2 = tlapack::row(A1, piv0[j]);
                tlapack::swap(vect1, vect2);
            }
        }

        // partition A into the following four blocks:
        auto A00 = tlapack::slice(A, tlapack::range<idx_t>(0, k0),
                                  tlapack::range<idx_t>(0, k0));
        auto A01 = tlapack::slice(A, tlapack::range<idx_t>(0, k0),
                                  tlapack::range<idx_t>(k0, n));
        auto A10 = tlapack::slice(A, tlapack::range<idx_t>(k0, m),
                                  tlapack::range<idx_t>(0, k0));
        auto A11 = tlapack::slice(A, tlapack::range<idx_t>(k0, m),
                                  tlapack::range<idx_t>(k0, n));

        // Take piv1 to be the second slice of of piv, meaning piv= [piv0, piv1]
        auto piv1 = tlapack::slice(piv, tlapack::range<idx_t>(k0, k));

        // Solve the triangular system of equations given by A00 X = A01
        trsm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::Unit, T(1), A00, A01);

        // A11 <---- A11 - (A10 * A01)
        gemm(Op::NoTrans, Op::NoTrans, real_t(-1), A10, A01, real_t(1), A11);

        // Finding LU factorization of A11 in place
        info = getrf_recursive(A11, piv1);
        if (info != 0) return info + k0;

        // swap the rows of A10 according to the swapped rows of A11 by refering
        // to piv1
        for (idx_t j = 0; j < k - k0; j++) {
            if (piv1[j] != j) {
                auto vect1 = tlapack::row(A10, j);
                auto vect2 = tlapack::row(A10, piv1[j]);
                tlapack::swap(vect1, vect2);
            }
        }

        // Shift piv1, so piv will have the accurate representation of overall
        // pivots
        for (idx_t i = 0; i < k - k0; i++) {
            piv1[i] += k0;
        }

        return 0;
    }

}  // getrf_recursive

}  // namespace tlapack

#endif  // TLAPACK_GETRF_RECURSIVE_HH
