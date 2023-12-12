/// @file gels.hpp
/// @author David Li, University of California, Berkeley
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgels.f
//
// Copyright (c) 2021-2023, University of California, Berkeley. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GELS_HH
#define TLAPACK_GELS_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/gelqf.hpp"
#include "tlapack/lapack/geqrf.hpp"
#include "tlapack/lapack/unmqr.hpp"
#include "tlapack/lapack/lascl.hpp"
#include "tlapack/lapack/laset.hpp"
#include "tlapack/blas/trsm.hpp"

namespace tlapack {
/**
 * @brief Solves overdetermined or underdetermined complex linear systems
 * 
 * @param[in] A m-by-n matrix.
 * 
 * @param[in,out] B On entry, the m-by-nrhs right hand side matrix B.
 * On Exit, if m >= n, B is overwritten by the n-by-nrhs solution matrix X;
 * 
 * 
 * *> ZGELS solves overdetermined or underdetermined complex linear systems
*> involving an M-by-N matrix A, or its conjugate-transpose, using a QR
*> or LQ factorization of A.  It is assumed that A has full rank.
*>
*> The following options are provided:
*>
*> 1. If TRANS = 'N' and m >= n:  find the least squares solution of
*>    an overdetermined system, i.e., solve the least squares problem
*>                 minimize || B - A*X ||.
*>
*> 2. If TRANS = 'N' and m < n:  find the minimum norm solution of
*>    an underdetermined system A * X = B.
*>
*> 3. If TRANS = 'C' and m >= n:  find the minimum norm solution of
*>    an underdetermined system A**H * X = B.
*>
*> 4. If TRANS = 'C' and m < n:  find the least squares solution of
*>    an overdetermined system, i.e., solve the least squares problem
*>                 minimize || B - A**H * X ||.
*/
template <TLAPACK_SMATRIX matrix_t>
int gels(matrix_t& A, matrix_t& B, Op& trans) {
    //using vector_t = vector_type<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    Create<vector_type<matrix_t>> new_vector;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t m_B = nrows(B);
    const idx_t n_B = ncols(B);

    if (m>=n) {
        // Calculate QR factorization of A
        // Create a new vector_t to store the tau values
        std::vector<type_t<matrix_t>> tauw_;
        auto tauw = new_vector(tauw_, k);
        geqrf(A, tauw);
        if (trans == Op::NoTrans) {
            // Solve overdetermined system
            // Apply Q^T to b
            //auto R = slice(A, range(0, k), range(0, k));
            unmqr(Side::Left, Op::ConjTrans, A, tauw, B);
            // Solve R*X = Q^T*b
            auto R = slice(A, range(0, n_B), range(0, n_B));
            B = slice(B, range(0, n), range(0, n_B));
            trsm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, real_t(1), R, B); // A should be n_B by n_B
        }
    }
    return 0; 
}

} // namespace tlapack
#endif
