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
#include "tlapack/blas/trmm.hpp"

namespace tlapack {

/// @brief Options struct for llh_mult()
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

    if (n <= 1) {
        // U(0, 0) = real(U(0,0))*real(U(0,0));
        U(0, 0) = conj(U(0, 0)) * U(0, 0);
        return;
    }

    if (n <= opts.nx) {
        for (idx_t j = n; j-- > 0; ) {
            real_t real_part_of_cjj;
            real_part_of_cjj = real(U(j, j)) * real(U(j, j));
            for(idx_t k = 0; k < j; ++k) {
                real_part_of_cjj += real(U(k, j)) * real(U(k, j)) + imag(U(k, j)) * imag(U(k, j));
            }
            U(j,j) = real_part_of_cjj;
            for (idx_t i = j; i-- > 0; ) {
                U(i, j) = conj(U(i, i)) * U(i, j);
                for (idx_t k = i; k-- > 0; ) {
                    U(i, j) += conj(U(k, i)) * U(k, j);
                }
            }
        }
        // for (idx_t i = n; i-- > 0;) {
        //     real_t sum(0);
        //     for (idx_t k = 0; k <= i; ++k) {
        //         // sum += C(i, k) * std::conj(C(i, k));
        //         sum += real(L(i, k)) * real(L(i, k)) +
        //                imag(L(i, k)) * imag(L(i, k));
        //     }
        //     L(i, i) = sum;

        //     for (idx_t j = i; j-- > 0;) {
        //         T sum(0);
        //         for (idx_t k = 0; k <= j; ++k) {
        //             sum += L(i, k) * conj(L(j, k));
        //         }
        //         L(i, j) = sum;
        //     }
        // }
    }

    // Recursive case: divide into blocks
    // const idx_t n0 = n / 2;

    // auto L00 = slice(L, range(0, n0), range(0, n0));
    // auto L10 = slice(L, range(n0, n), range(0, n0));
    // auto L11 = slice(L, range(n0, n), range(n0, n));

    // // L11 = L11*L11^H
    // mult_llh(L11, opts);

    // // L11 += L10 * L10^H
    // herk(Uplo::Lower, Op::NoTrans, real_t(1), L10, real_t(1), L11);

    // A10 = A10 * A00^H
    // trmm(Side::Right, Uplo::Lower, Op::ConjTrans, Diag::NonUnit, T(1), L00,
    //      L10);

    // // A00 = A00 * A00^H
    // mult_llh(L00, opts);


    const idx_t n0 = n / 2;

    auto U00 = slice(U, range(0, n0), range(0, n0));
    auto U01 = slice(U, range(0, n0), range(n0, n));
    auto U11 = slice(U, range(n0, n), range(n0, n));

    // U11 = U11^H*U11
    mult_uhu(U11, opts);

    // U11+= U01^H*U01
    herk(Uplo::Upper, Op::ConjTrans, real_t(1),  U01, real_t(1), U11);

    // U01 = U00^H*U01
    trmm(Side::Left, Uplo::Upper, Op::ConjTrans, Diag::NonUnit, real_t(1), U00, U01);

    // U00 = U00^H*U00
    mult_uhu(U00, opts);    
    return;
}

}  // namespace tlapack

#endif  // TLAPACK_MULT_LLH
