/// @file mult_lhl_oop.hpp
/// @author Kyle Cunningham, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MULT_LHL_OOP
#define TLAPACK_MULT_LHL_OOP

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/trmm.hpp"

namespace tlapack {

/// @brief Options struct for mult_lhl_oop()
struct mult_lhl_oop_Opts {
    /// Optimization parameter. Matrices smaller than nx will not
    /// be multiplied using recursion. Must be at least 1.
    size_t nx = 1;
};

/**
 *
 * @brief TODO
 *
 * @param[in] L n-by-n matrix
 * On entry, the lower triangular matrix L. On exit, unchanged.
 *
 * @param[in/out] A n-by-n matrix On entry, the Hermitian matrix A. On exit, A
 * contains the updated uplo part of the Hermitian product L^H*L. The not uplo
 * triangular entries of L are not referenced.
 *
 * TODO: we want uplo for A and L, we want trans for L, we want alpha and beta
 * we just do what we need today. We need
 *       A ⟵   - L^H L + A
 * where L is lower and A is upper
 *
 * @param[in] opts Options.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SMATRIX matrix_t>
void mult_lhl_oop_cheat(matrix_t& L,
                        matrix_t& A,
                        const mult_lhl_oop_Opts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    tlapack::Create<matrix_t> new_matrix;

    const idx_t n = nrows(L);
    // tlapack_check(n == ncols(L));
    // tlapack_check(opts.nx >= 1);

    // Quick return
    if (n == 0) return;

    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);

    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < j; i++)
            work(i, j) = real_t(0);
        for (idx_t i = j; i < n; i++)
            work(i, j) = L(i, j);
    }

    herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans, real_t(-1), work,
         real_t(1), A);

    return;
}
template <TLAPACK_SMATRIX matrix_t>
void mult_lhl_oop(matrix_t& L, matrix_t& A, const mult_lhl_oop_Opts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    tlapack_check(nrows(L) == ncols(L));
    tlapack_check(nrows(A) == ncols(A));
    tlapack_check(nrows(L) == nrows(A));

    const idx_t n = nrows(L);
    // Quick return
    if (n == 0) return;

    const idx_t nx = std::min<idx_t>(opts.nx, n);  // block size

    if (n <= nx) {
        // only works for nx = 1 for now
        // or we can use the "cheat" if nx is not 1
        //
        // base case
        // Lower-triangular herk: A = -L^H L + A
        //
        // mult_lhl_oop_cheat(L, A, opts);
        //
        A(0, 0) -=
            (real(L(0, 0)) * real(L(0, 0)) + imag(L(0, 0)) * imag(L(0, 0)));
        return;
    }

    idx_t n1 = n / 2;
    idx_t n2 = n - n1;

    auto L00 = slice(L, range(0, n1), range(0, n1));
    auto L10 = slice(L, range(n1, n), range(0, n1));
    auto L11 = slice(L, range(n1, n), range(n1, n));

    auto A00 = slice(A, range(0, n1), range(0, n1));
    auto A01 = slice(A, range(0, n1), range(n1, n));
    auto A11 = slice(A, range(n1, n), range(n1, n));

    mult_lhl_oop(L00, A00, opts);

    herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans, real_t(-1), L10,
         real_t(1), A00);

    trmm_out(Side::Right, Uplo::Lower, Op::NoTrans, Diag::NonUnit,
             Op::ConjTrans, real_t(-1), L11, L10, real_t(1), A01);

    mult_lhl_oop(L11, A11), opts;

    return;
}

}  // namespace tlapack

#endif  // TLAPACK_MULT_LHL_OOP
