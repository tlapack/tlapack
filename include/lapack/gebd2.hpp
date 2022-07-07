/// @file gebd2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgebd2.f
//
// Copyright (c) 2014-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_GEBD2_HH__
#define __TLAPACK_GEBD2_HH__

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace tlapack
{

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
     * @param work Vector of size max(m,n).
     *
     * @ingroup gebrd
     */
    template <typename matrix_t, class vector_t>
    int gebd2(matrix_t &A, vector_t &tauv, vector_t &tauw, vector_t &work)
    {

        using idx_t = size_type<matrix_t>;
        using T = type_t<matrix_t>;
        using range = std::pair<idx_t, idx_t>;

        // constants
        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        // check arguments
        tlapack_check_false(access_denied(dense, write_policy(A)), -1);
        tlapack_check_false((idx_t)size(tauv) < std::min<idx_t>(m, n), -2);
        tlapack_check_false((idx_t)size(tauw) < std::min<idx_t>(m, n), -3);
        tlapack_check_false((idx_t)size(work) < std::max<idx_t>(m, n), -4);

        // quick return
        if (n <= 0)
            return 0;

        for (idx_t j = 0; j < n; ++j)
        {

            // Generate elementary reflector H(j) to annihilate A(j+1:m,j)
            auto v = slice(A, range(j, m), j);
            larfg(v, tauv[j]);

            // Apply H(j)**H to A(j:m,j+1:n) from the left
            if (j < n - 1)
            {
                auto A11 = slice(A, range(j, m), range(j + 1, n));
                larf(Side::Left, v, conj(tauv[j]), A11, work);
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
                    larf(Side::Right, w, tauw[j], B11, work);
                }
                for (idx_t i = 0; i < n-j-1; ++i) 
                    w[i] = conj(w[i]);
            }
        }

        return 0;
    }
}
#endif // __TLAPACK_GEBD2_HH__