/// @file ungl2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zungl2.f
//
// Copyright (c) 2014-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_UNGL2_HH__
#define __TLAPACK_UNGL2_HH__

#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace tlapack
{
    /**
     * Generates all or part of the unitary matrix Q from an LQ factorization
     * determined by gelq2 (unblocked algorithm).
     *
     * The matrix Q is defined as the first k rows of a product of k elementary
     * reflectors of order n
     * \[
     *          Q = H(k)**H ... H(2)**H H(1)**H
     * \]
     * as returned by gelq2 and k <= n. 
     *
     * @return  0 if success
     *
     * @param[in,out] Q k-by-n matrix.
     *      On entry, the i-th row must contain the vector which defines
     *      the elementary reflector H(j), for j = 1,2,...,k, as returned
     *      by gelq2 in the first k rows of its array argument A.
     *      On exit, the k by n matrix Q.
     *
     * @param[out] tauw Complex vector of length min(m,n).
     *      tauw(j) must contain the scalar factor of the elementary
     *      reflector H(j), as returned by gelq2.
     *
     * @param work Vector of size k.
     *
     * @ingroup ungl2
     */
    template <typename matrix_t, class vector_t>
    int ungl2(matrix_t &Q, vector_t &tauw, vector_t &work)
    {
        using idx_t = size_type<matrix_t>;
        using T = type_t<matrix_t>;
        using range = std::pair<idx_t, idx_t>;

        // constants
        const idx_t k = nrows(Q);
        const idx_t n = ncols(Q);
        const idx_t m = size(tauw);     // maximum number of Householder reflectors to use
        const idx_t t = min(k, m);      // desired number of Householder reflectors to use

        // check arguments
        tlapack_check_false(access_denied(dense, write_policy(Q)), -1);
        tlapack_check_false((idx_t)size(tauw) < std::min<idx_t>(m, n), -2);
        tlapack_check_false((idx_t)size(work) < t, -3);

        // Initialise matrix C with 1 on its diagonal 
        std::vector<T> C_(k * n);
        legacyMatrix<T> C(k, n, &C_[0], k);
        laset(Uplo::General, T(0), T(1), C);

        for (idx_t j = t - 1; j != -1; --j)
        {   
            // Apply H(j)**H to Q(j:k,j:n) from the right
            auto w = slice(Q, j, range(j, n));
            for (idx_t i = 0; i < n - j; ++i)
                w[i] = conj(w[i]);

            auto C11 = slice(C, range(0, k), range(j, n));

            larf(Side::Right, w, conj(tauw[j]), C11, work);
            for (idx_t i = 0; i < n - j; ++i)
                w[i] = conj(w[i]);
        }

        lacpy(Uplo::General, C, Q);

        return 0;
    }
}
#endif // __TLAPACK_UNGL2_HH__