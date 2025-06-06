/// @file mult_uhu.hpp
/// @author Ella Addison-Taylor, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MULT_UHU
#define TLAPACK_MULT_UHU

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trmm.hpp"

namespace tlapack {

/// @brief Options struct for mult_uhu()
struct mult_uhu_Opts {
    /// Optimization parameter. Matrices smaller than nx will not
    /// be multiplied using recursion. Must be at least 1.
    size_t nx = 1;
};

/**
 *
 * @brief in-place multiplication of upper triangular matrix U and lower
 * triangular matrix U^H. This is the recursive variant.
 *
 * @param[in,out] U n-by-n matrix
 *      On entry, the upper triangular matrix U. On exit, U contains the upper
 * part of the Hermitian product U^H*U. The lower triangular entries of U are
 * not referenced.
 *
 * @param[in] opts Options.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SMATRIX matrix_t>
void mult_uhu(matrix_t& U, const mult_uhu_Opts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(U);
    tlapack_check(n == ncols(U));
    tlapack_check(opts.nx >= 1);

    // quick return
    if (n == 0) return;

    if (n <= opts.nx) {
        for (idx_t j = n; j-- > 0;) {
            T real_part_of_ujj;
            real_part_of_ujj =
                real(U(j, j)) * real(U(j, j)) + imag(U(j, j)) * imag(U(j, j));
            for (idx_t k = 0; k < j; ++k) {
                real_part_of_ujj += real(U(k, j)) * real(U(k, j)) +
                                    imag(U(k, j)) * imag(U(k, j));
            }
            U(j, j) = real_part_of_ujj;
            for (idx_t i = j; i-- > 0;) {
                U(i, j) = conj(U(i, i)) * U(i, j);
                for (idx_t k = i; k-- > 0;) {
                    U(i, j) += conj(U(k, i)) * U(k, j);
                }
            }
        }
        return;
    }

    const idx_t n0 = n / 2;

    auto U00 = slice(U, range(0, n0), range(0, n0));
    auto U01 = slice(U, range(0, n0), range(n0, n));
    auto U11 = slice(U, range(n0, n), range(n0, n));

    // U11 = U11^H*U11
    mult_uhu(U11, opts);

    // U11+= U01^H*U01
    herk(Uplo::Upper, Op::ConjTrans, real_t(1), U01, real_t(1), U11);

    // U01 = U00^H*U01
    trmm(Side::Left, Uplo::Upper, Op::ConjTrans, Diag::NonUnit, real_t(1), U00,
         U01);

    // U00 = U00^H*U00
    mult_uhu(U00, opts);

    return;
}

}  // namespace tlapack

#endif  // TLAPACK_MULT_UHU