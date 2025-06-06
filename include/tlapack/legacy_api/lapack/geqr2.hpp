/// @file legacy_api/lapack/geqr2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/geqr2.h
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_GEQR2_HH
#define TLAPACK_LEGACY_GEQR2_HH

#include "tlapack/lapack/geqr2.hpp"

namespace tlapack {
namespace legacy {

    /** Computes a QR factorization of a matrix A.
     *
     * @param[in] m The number of rows of the matrix A.
     * @param[in] n The number of columns of the matrix A.
     * @param[in,out] A m-by-n matrix.
     *      On exit, the elements on and above the diagonal of the array
     *      contain the min(m,n)-by-n upper trapezoidal matrix R
     *      (R is upper triangular if m >= n); the elements below the diagonal,
     *      with the array tau, represent the unitary matrix Q as a
     *      product of elementary reflectors.
     * @param[in] lda The leading dimension of A. lda >= max(1,m).
     * @param[out] tau Real vector of length min(m,n).
     *      The scalar factors of the elementary reflectors.
     *      The subarray tau[1:n-1] is used as the workspace.
     *
     * @see geqr2( matrix_t& A, vector_t &tau, vector_t &work )
     *
     * @ingroup legacy_lapack
     */
    template <typename TA, typename TT>
    int geqr2(idx_t m, idx_t n, TA* A, idx_t lda, TT* tau)
    {
        using internal::create_matrix;
        using internal::create_vector;

        // check arguments
        tlapack_check_false(m < 0);
        tlapack_check_false(n < 0);
        tlapack_check_false(lda < m);

        // quick return
        if (n <= 0) return 0;

        // Matrix views
        auto A_ = create_matrix(A, m, n, lda);
        auto tau_ = create_vector(tau, min(m, n));

        return geqr2(A_, tau_);
    }

}  // namespace legacy
}  // namespace tlapack

#endif  // TLAPACK_LEGACY_GEQR2_HH
