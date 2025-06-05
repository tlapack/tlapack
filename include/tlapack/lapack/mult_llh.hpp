/// @file lu_mult.hpp
/// @author Lindsay Slager, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MULT_LLH
#define TLAPACK_MULT_LLH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/trmm.hpp"

namespace tlapack {

/// @brief Options struct for lu_mult()
struct LuMultOpts {
    /// Optimization parameter. Matrices smaller than nx will not
    /// be multiplied using recursion. Must be at least 1.
    size_t nx = 1;
};

/**
 *
 * @brief in-place multiplication of lower triangular matrix L and upper
 * triangular matrix U. this is the recursive variant
 *
 * @param[in,out] C n-by-n matrix
 *      On entry, the strictly lower triangular entries of A contain the matrix
 * L. L is assumed to have unit diagonal. The upper triangular entires of A
 * contain the matrix U. On exit, A contains the product L*U.
 *
 * @param[in] opts Options.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SMATRIX matrix_t>
void mult_llh(matrix_t& C, const LuMultOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(C);
    tlapack_check(n == ncols(C));
    tlapack_check(opts.nx >= 1);

    if (n <= 1)
    {
        C(0, 0) = C(0, 0) * conj(C(0, 0));
        return;
    }

    if (n <= opts.nx)
    {
        for (idx_t i = n; i-- > 0;) {
            real_t sum(0);
            for (idx_t k = 0; k <= i; ++k) {
                //sum += C(i, k) * std::conj(C(i, k));
                sum += real(C(i, k)) * real(C(i, k)) + imag(C(i,k)) * imag(C(i,k));
            }
            C(i, i) = sum;

            for (idx_t j = i; j-- > 0;) {
                T sum(0);
                for (idx_t k = 0; k <= j; ++k) {
                    sum += C(i, k) * conj(C(j, k));
                }
                C(i, j) = sum;
            }
        }
    }

    // Recursive case: divide into blocks
    const idx_t n0 = n / 2;

    auto C00 = slice(C, range(0, n0), range(0, n0));
    auto C10 = slice(C, range(n0, n), range(0, n0));
    auto C11 = slice(C, range(n0, n), range(n0, n));

    // A11 = A11*A11^H
    mult_llh(C11, opts);

    // A11 += A10 * A10^H
    herk(Uplo::Lower, Op::NoTrans, real_t(1), C10, real_t(1), C11);

    // A10 = A10 * A00^H
    trmm(Side::Right, Uplo::Lower, Op::ConjTrans, Diag::NonUnit, T(1), C00, C10);

    // A00 = A00 * A00^H
    mult_llh(C00, opts);

    return;
}

}  // namespace tlapack

#endif  // TLAPACK_MULT_LLH
