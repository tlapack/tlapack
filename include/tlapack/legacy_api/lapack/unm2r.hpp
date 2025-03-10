/// @file legacy_api/lapack/unm2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/ormr2.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_UNM2R_HH
#define TLAPACK_LEGACY_UNM2R_HH

#include "tlapack/lapack/unm2r.hpp"

namespace tlapack {
namespace legacy {

    /** Applies unitary matrix Q to a matrix C.
     *
     * @return  0 if success
     * @return -i if the ith argument is invalid
     *
     * @param[in] side Specifies which side Q is to be applied.
     *                 'L': apply Q or Q' from the Left;
     *                 'R': apply Q or Q' from the Right.
     * @param[in] trans Specifies whether Q or Q' is applied.
     *                 'N':  No transpose, apply Q;
     *                 'T':  Transpose, apply Q'.
     * @param[in] m The number of rows of the matrix C.
     * @param[in] n The number of columns of the matrix C.
     * @param[in] k The number of elementary reflectors whose product defines
     * the matrix Q. If side='L', m>=k>=0; if side='R', n>=k>=0.
     * @param[in] A Matrix containing the elementary reflectors H.
     *                 If side='L', A is k-by-m;
     *                 if side='R', A is k-by-n.
     * @param[in] lda The column length of the matrix A.  ldA>=k.
     * @param[in] tau Real vector of length k containing the scalar factors of
     * the elementary reflectors.
     * @param[in,out] C m-by-n matrix.
     *     On exit, C is replaced by one of the following:
     *                 If side='L' & trans='N':  C <- Q * C
     *                 If side='L' & trans='T':  C <- Q'* C
     *                 If side='R' & trans='T':  C <- C * Q'
     *                 If side='R' & trans='N':  C <- C * Q
     * @param ldc The column length the matrix C. ldC>=m.
     *
     * @ingroup legacy_lapack
     */
    template <class side_t, class trans_t, typename TA, typename TC>
    int unm2r(side_t side,
              trans_t trans,
              idx_t m,
              idx_t n,
              idx_t k,
              TA* A,
              idx_t lda,
              const real_type<TA, TC>* tau,
              TC* C,
              idx_t ldc)
    {
        // check arguments
        tlapack_check_false(side != Side::Left && side != Side::Right);
        tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                            trans != Op::ConjTrans);
        tlapack_check_false(m < 0);
        tlapack_check_false(n < 0);
        const idx_t q = (side == Side::Left) ? m : n;
        tlapack_check_false(k < 0 || k > q);
        tlapack_check_false(lda < q);
        tlapack_check_false(ldc < m);

        // quick return
        if ((m == 0) || (n == 0) || (k == 0)) return 0;

        // Matrix views
        const auto A_ = create_matrix<TA>((TA*)A, q, k, lda);
        const auto tau_ = create_vector((TA*)tau, k);
        auto C_ = create_matrix<TC>(C, m, n, ldc);

        return unm2r(side, trans, A, tau_, C_);
    }

}  // namespace legacy
}  // namespace tlapack

#endif  // TLAPACK_LEGACY_UNM2R_HH