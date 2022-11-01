/// @file ungl2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zungl2.f
//
// Copyright (c) 2014-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGL2_HH
#define TLAPACK_UNGL2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack
{

    /** Worspace query.
     * @see ungl2
     * 
     * @param[out] workinfo On return, contains the required workspace sizes.
     */
    template< class matrix_t, class vector_t >
    inline constexpr
    void ungl2_worksize(
        matrix_t& Q, vector_t &tauw, workinfo_t& workinfo,
        const workspace_opts_t<>& opts = {} )
    {
        using idx_t = size_type< matrix_t >;

        // constants
        const idx_t k = nrows(Q);

        if( k > 1 ) {
            auto C = rows( Q, range<idx_t>{1,k} );
            larf_worksize( right_side, row(Q,0), tauw[0], C, workinfo, opts );
        }
        else
            workinfo = {};
    }
    
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
     * @param[in] tauw Complex vector of length min(m,n).
     *      tauw(j) must contain the scalar factor of the elementary
     *      reflector H(j), as returned by gelq2.
     *
    * @param[in] opts Options.
     *      @c opts.work is used if whenever it has sufficient size.
     *      The sufficient size can be obtained through a workspace query.
     *
     * @ingroup ungl2
     */
    template <typename matrix_t, class vector_t>
    int ungl2(matrix_t &Q, const vector_t &tauw, const workspace_opts_t<>& opts = {})
    {
        using idx_t = size_type<matrix_t>;
        using T = type_t<matrix_t>;
        using range = std::pair<idx_t, idx_t>;
        using real_t = real_type<T>;

        // constants
        const idx_t k = nrows(Q);
        const idx_t n = ncols(Q);
        const idx_t m = size(tauw); // maximum number of Householder reflectors to use
        const idx_t t = min(k, m);  // desired number of Householder reflectors to use

        // check arguments
        tlapack_check_false(access_denied(dense, write_policy(Q)) );
        tlapack_check_false((idx_t)size(tauw) < std::min<idx_t>(m, n) );

        // Allocates workspace
        vectorOfBytes localworkdata;
        Workspace work = [&]()
        {
            workinfo_t workinfo;
            ungl2_worksize( Q, tauw, workinfo, opts );
            return alloc_workspace( localworkdata, workinfo, opts.work );
        }();
        
        // Options to forward
        auto&& larfOpts = workspace_opts_t<>{ work };

        // Initialise columns t:k-1 to rows of the unit matrix
        if (k > m)
        {
            for (idx_t j = 0; j < n; ++j)
            {
                for (idx_t i = t; i < k; ++i)
                    Q(i, j) = make_scalar<T>(0, 0);
                if (j < k && j > t - 1)
                    Q(j, j) = make_scalar<T>(1, 0);
            }
        }

        for (idx_t j = t - 1; j != idx_t(-1); --j)
        {
            // Apply H(j)**H to Q(j:k,j:n) from the right
            if (j + 1 < n)
            {
                auto w = slice(Q, j, range(j, n));

                // Apply to the Q11 below the w
                if( (k > m && j + 1 == t) || (j + 1 < t) )
                {
                    // When k > m, we need to start from the (t-1)th row of w and apply to Q(j+1:k,j:n)
                    // This procedure is only used once when both conditions are satisfied
                    
                    for (idx_t i = 0; i < n - j; ++i)
                        w[i] = conj(w[i]);

                    Q(j, j) = make_scalar<T>(1, 0);

                    auto Q11 = slice(Q, range(j + 1, k), range(j, n));
                    larf(Side::Right, w, conj(tauw[j]), Q11, larfOpts);
                    
                    for (idx_t i = 0; i < n - j; ++i)
                        w[i] = conj(w[i]);
                }

                scal(-conj(tauw[j]), w);
            }

            Q(j, j) = real_t(1.) - conj(tauw[j]);

            // Set Q(j,0:j-1) to zero
            for (idx_t l = 0; l < j; l++)
                Q(j, l) = make_scalar<T>(0, 0);
        }

        return 0;
    }
}
#endif // TLAPACK_UNGL2_HH
