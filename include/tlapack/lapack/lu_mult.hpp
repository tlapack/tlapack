/// @file lu_mult.hpp
/// @author Lindsay Slager, University of Colorado Denver, USA
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LU_MULT_HH
#define TLAPACK_LU_MULT_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/trmm.hpp"

namespace tlapack {

template <typename idx_t>
struct lu_mult_opts_t
{
    // Optimization parameter. Matrices smaller than nx will not
    // be multiplied using recursion. Must be at least 1.
    idx_t nx = 1;
};

/**
 *
 * @brief in-place multiplication of lower triangular matrix L and upper triangular matrix U.
 *      this is the recursive variant
 * 
 * @param[in,out] A n-by-n matrix
 *      On entry, the strictly lower triangular entries of A contain the matrix L.
 *      L is assumed to have unit diagonal.  
 *      The upper triangular entires of A contain the matrix U. 
 *      On exit, A contains the product L*U. 
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 * 
 * @ingroup auxiliary
 */
template <class matrix_t>
void lu_mult(matrix_t &A, const lu_mult_opts_t<size_type<matrix_t>> &opts = {})
{
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    using real_t = real_type<T>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    tlapack_check(m == n);
    tlapack_check(opts.nx >= 1);

    // quick return
    if (n == 0)
        return;

    if (n <= opts.nx)
    { // Matrix is small, use for loops instead of recursion
        for (idx_t i2 = n; i2 > 0; --i2)
        {
            idx_t i = i2 - 1;
            for (idx_t j2 = n; j2 > 0; --j2)
            {
                idx_t j = j2 - 1;
                T sum( 0 );
                for (idx_t k = 0; k <= min(i, j); ++k)
                {
                    if (i == k)
                        sum += A(k, j);
                    else
                        sum += A(i, k) * A(k, j);
                }
                A(i, j) = sum;
            }
        }
        return;
    }

    const idx_t n0 = n / 2;

    /*
        Matrix A is splitted into 4 submatrices:
        A = [ A00 A01 ]
            [ A10 A11 ]
        and, hereafter,
            L00 is the strict lower triangular part of A00, with unitary main diagonal.
            L11 is the strict lower triangular part of A11, with unitary main diagonal.
            U00 is the upper triangular part of A00.
            U11 is the upper triangular part of A11.
    */
    auto A00 = slice(A, range(0, n0), range(0, n0));
    auto A01 = slice(A, range(0, n0), range(n0, n));
    auto A10 = slice(A, range(n0, n), range(0, n0));
    auto A11 = slice(A, range(n0, n), range(n0, n));

    lu_mult(A11, opts);

    // A11 = A10*A01 + L11*U11
    gemm(Op::NoTrans, Op::NoTrans, T(1), A10, A01, T(1), A11);

    // A01 = L00*A01
    trmm(Side::Left, Uplo::Lower, Op::NoTrans,
                  Diag::Unit, real_t(1), A00, A01);

    // A10 = A10*U00
    trmm(Side::Right, Uplo::Upper, Op::NoTrans,
                  Diag::NonUnit, real_t(1), A00, A10);

    // A00 = L00*U00
    lu_mult(A00, opts);

    return;
}

}

#endif // TLAPACK_LU_MULT_HH
