/// @file gebd2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgebd2.f
//
// Copyright (c) 2014-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEBD2_HH
#define TLAPACK_GEBD2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack
{

    struct gebd2_opts_t : public workspace_opts_t<>
    {
        inline constexpr gebd2_opts_t( const workspace_opts_t<>& opts = {} )
        : workspace_opts_t<>( opts ) {};
    };

    /** Worspace query.
     * @see gebd2
     * 
     * @param[out] workinfo On return, contains the required workspace sizes.
     */
    template <typename matrix_t, class vector_t>
    inline constexpr 
    void gebd2_worksize(matrix_t &A, vector_t &tauv, vector_t &tauw, workinfo_t& workinfo, const gebd2_opts_t& opts = {})
    {
        using idx_t = size_type< matrix_t >;

        // constants
        const idx_t m = nrows(A);
        const idx_t n = ncols(A);
        
        if( n > 1 )
        {
            auto A11 = cols( A, range<idx_t>{1,n} );
            larf_worksize( left_side, col(A,0), tauv[0], A11, workinfo, opts );
            
            if( m > 1 )
            {
                workinfo_t workinfo2;
                auto B11 = rows( A11, range<idx_t>{1,m} );
                auto row0 = slice(A,0,range<idx_t>{1,n});
                
                larf_worksize( right_side, row0, tauw[0], B11, workinfo2, opts );
                
                workinfo.minMax( workinfo2 );
            }
        }
        else
            workinfo = {};
    }

    /** Reduces a complex general m by n matrix A to an upper
     *  real bidiagonal form B by a unitary transformation:
     * \[
     *          Q**H * A * Z = B,
     * \]
     *  where m >= n.
     *
     * The matrices Q and Z are represented as products of elementary
     * reflectors:
     *
     * If m >= n,
     * \[
     *          Q = H(1) H(2) . . . H(n)  and  Z = G(1) G(2) . . . G(n-1)
     * \]
     * Each H(i) and G(i) has the form:
     * \[
     *          H(j) = I - tauv * v * v**H  and G(j) = I - tauw * w * w**H
     * \]
     * where tauv and tauw are complex scalars, and v and w are complex
     * vectors; v(1:j-1) = 0, v(j) = 1, and v(j+1:m) is stored on exit in
     * A(j+1:m,j); w(1:j) = 0, w(j+1) = 1, and w(j+2:n) is stored on exit in
     * A(j,i+2:n); tauv is stored in tauv(j) and tauw in tauw(j).
     *
     * @return  0 if success
     *
     * @param[in,out] A m-by-n matrix.
     *      On entry, the m by n general matrix to be reduced.
     *      On exit, if m >= n, the diagonal and the first superdiagonal
     *      are overwritten with the upper bidiagonal matrix B; the
     *      elements below the diagonal, with the array tauv, represent
     *      the unitary matrix Q as a product of elementary reflectors,
     *      and the elements above the first superdiagonal, with the array
     *      tauw, represent the unitary matrix Z as a product of elementary
     *      reflectors.
     *
     * @param[out] tauv Real vector of length min(m,n).
     *      The scalar factors of the elementary reflectors which
     *      represent the unitary matrix Q.
     *
     * @param[out] tauw Real vector of length min(m,n).
     *      The scalar factors of the elementary reflectors which
     *      represent the unitary matrix Z.
     *
     * @param[in] opts Options.
     *      - @c opts.work is used if whenever it has sufficient size.
     *        The sufficient size can be obtained through a workspace query.
     *
     * @ingroup gebrd
     */
    template <typename matrix_t, class vector_t>
    int gebd2(matrix_t &A, vector_t &tauv, vector_t &tauw, const gebd2_opts_t& opts = {})
    {

        using idx_t = size_type<matrix_t>;
        using range = std::pair<idx_t, idx_t>;

        // constants
        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        // check arguments
        tlapack_check_false(access_denied(dense, write_policy(A)));
        tlapack_check_false((idx_t)size(tauv) < std::min<idx_t>(m, n));
        tlapack_check_false((idx_t)size(tauw) < std::min<idx_t>(m, n));
        tlapack_check(m >= n); // Only m >= n matrices are supported (yet).

        // quick return
        if (n <= 0)
            return 0;

        // Allocates workspace
        vectorOfBytes localworkdata;
        Workspace work = [&]()
        {
            workinfo_t workinfo;
            gebd2_worksize( A, tauv, tauw, workinfo, opts );
            return alloc_workspace( localworkdata, workinfo, opts.work );
        }();
        
        // Options to forward
        auto&& larfOpts = workspace_opts_t<>{ work };

        for (idx_t j = 0; j < n; ++j)
        {

            // Generate elementary reflector H(j) to annihilate A(j+1:m,j)
            auto v = slice(A, range(j, m), j);
            larfg(v, tauv[j]);

            // Apply H(j)**H to A(j:m,j+1:n) from the left
            if (j < n - 1)
            {
                auto A11 = slice(A, range(j, m), range(j + 1, n));
                larf(Side::Left, v, conj(tauv[j]), A11, larfOpts);
            }

            if (j < n - 1)
            {
                // Generate elementary reflector G(j) to annihilate A(j,j+2:n)
                auto w = slice(A, j, range(j + 1, n));
                for (idx_t i = 0; i < n - j - 1; ++i)
                    w[i] = conj(w[i]);
                larfg(w, tauw[j]);

                // Apply G(j) to A(j+1:m,j+1:n) from the right
                if (j < m - 1)
                {
                    auto B11 = slice(A, range(j + 1, m), range(j + 1, n));
                    larf(Side::Right, w, tauw[j], B11, larfOpts);
                }
                for (idx_t i = 0; i < n-j-1; ++i) 
                    w[i] = conj(w[i]);
            }
        }

        return 0;
    }
}

#endif // TLAPACK_GEBD2_HH
