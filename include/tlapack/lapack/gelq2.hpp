/// @file gelq2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgelq2.f
//
// Copyright (c) 2014-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GELQ2_HH
#define TLAPACK_GELQ2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack
{

    /** Worspace query.
     * @see gelq2
     * 
     * @param[in,out] workinfo
     *      On output, the amount workspace required. It is larger than or equal
     *      to that given on input.
     */
    template< class matrix_t, class vector_t >
    inline constexpr
    void gelq2_worksize(
        matrix_t& A, vector_t &tauw, workinfo_t& workinfo,
        const workspace_opts_t<>& opts = {} )
    {
        using idx_t = size_type< matrix_t >;

        // constants
        const idx_t m = nrows(A);

        if( m > 1 ) {
            auto C = rows( A, range<idx_t>{1,m} );
            larf_worksize( right_side, forward, row(A,0), tauw[0], C, workinfo, opts );
        }
    }
    
    /** Computes an LQ factorization of a complex m-by-n matrix A using
     *  an unblocked algorithm.
     *
     * The matrix Q is represented as a product of elementary reflectors.
     * \[
     *          Q = H(k)**H ... H(2)**H H(1)**H,
     * \]
     * where k = min(m,n). Each H(j) has the form
     * \[
     *          H(j) = I - tauw * w * w**H
     * \]
     * where tauw is a complex scalar, and w is a complex vector with
     * \[
     *          w[0] = w[1] = ... = w[j-1] = 0; w[j] = 1,
     * \]
     * with w[j+1]**H through w[n]**H is stored on exit
     * in the jth row of A, and tauw in tauw[j].
     * 
     * 
     *
     * @return  0 if success
     *
     * @param[in,out] A m-by-n matrix.
     *      On exit, the elements on and below the diagonal of the array
     *      contain the m by min(m,n) lower trapezoidal matrix L (L is
     *      lower triangular if m <= n); the elements above the diagonal,
     *      with the array tauw, represent the unitary matrix Q as a
     *      product of elementary reflectors.
     *
     * @param[out] tauw Complex vector of length min(m,n).
     *      The scalar factors of the elementary reflectors.
     *
     * @param[in] opts Options.
     *      - @c opts.work is used if whenever it has sufficient size.
     *        The sufficient size can be obtained through a workspace query.
     *
     * @ingroup gelqf
     */
    template <typename matrix_t, class vector_t>
    int gelq2(matrix_t &A, vector_t &tauw, const workspace_opts_t<>& opts = {})
    {
        using idx_t = size_type<matrix_t>;
        using range = std::pair<idx_t, idx_t>;

        // constants
        const idx_t m = nrows(A);
        const idx_t n = ncols(A);
        const idx_t k = min(m, n);

        // check arguments
        tlapack_check_false(access_denied(dense, write_policy(A)) );
        tlapack_check_false((idx_t)size(tauw) < std::min<idx_t>(m, n) );

        // Allocates workspace
        vectorOfBytes localworkdata;
        Workspace work = [&]()
        {
            workinfo_t workinfo;
            gelq2_worksize( A, tauw, workinfo, opts );
            return alloc_workspace( localworkdata, workinfo, opts.work );
        }();
        
        // Options to forward
        auto&& larfOpts = workspace_opts_t<>{ work };

        for (idx_t j = 0; j < k; ++j)
        {
            // Define w := A(j,j:n)
            auto w = slice(A, j, range(j, n));

            // Generate elementary reflector H(j) to annihilate A(j,j+1:n)
            for (idx_t i = 0; i < n - j; ++i)
                w[i] = conj(w[i]); 
            larfg(w, tauw[j]);

            // If either condition is satisfied, Q11 will not be empty
            if (j < k - 1 || k < m)
            {
                // Apply H(j) to A(j+1:m,j:n) from the right
                auto Q11 = slice(A, range(j + 1, m), range(j, n));
                larf(Side::Right, forward, w, tauw[j], Q11, larfOpts);
            }
            for (idx_t i = 0; i < n - j; ++i)
                w[i] = conj(w[i]); 
        }

        return 0;
    }
}

#endif // TLAPACK_GELQ2_HH