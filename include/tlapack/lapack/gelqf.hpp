/// @file gelqf.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgelqf.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GELQF_HH
#define TLAPACK_GELQF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/gelq2.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"

namespace tlapack {
/**
 * Options struct for gelqf
 */
template <class matrix_t, class idx_t = size_t>
struct gelqf_opts_t : public workspace_opts_t<> {
    inline constexpr gelqf_opts_t(const workspace_opts_t<>& opts = {})
        : workspace_opts_t<>(opts){};

    idx_t nb = 32;  ///< Block size
    matrix_t* TT = NULL; // m-by-nb Matrix to store the triangular factors
};

/** Worspace query of gelqf()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param tau min(n,m) vector.
 *
 * @param[in] opts Options.
 *
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template <typename matrix_t,typename vector_t>
inline constexpr void gelqf_worksize(
    const matrix_t& A,
    const vector_t& tau,
    workinfo_t& workinfo,
    const gelqf_opts_t<matrix_t, size_type<matrix_t> >& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t nb = opts.nb;
    const idx_t ib = std::min<idx_t>(nb, k);

    auto A11 = rows(A, range<idx_t>(0, ib));
    auto A12 = slice(A, range<idx_t>(ib, m), range<idx_t>(0, n));

    gelq2_worksize(A11, tau, workinfo, opts);
    if( opts.TT == NULL ){
        workinfo_t workinfo2(sizeof(T)*m,nb);
        workinfo += workinfo2;
    }
}

/** Computes an LQ factorization of an m-by-n matrix A using
 *  a blocked algorithm.
 *
 * The matrix Q is represented as a product of elementary reflectors.
 * \[
 *          Q = H(k)**H ... H(2)**H H(1)**H,
 * \]
 * where k = min(m,n). Each H(j) has the form
 * \[
 *          H(j) = I - tauw * w * w**H
 * \]
 * where tauw is a scalar, and w is a vector with
 * \[
 *          w[0] = w[1] = ... = w[j-1] = 0; w[j] = 1,
 * \]
 * where w[j+1]**H through w[n]**H are stored on exit in the jth row of A.
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
 * @param[out] tau min(n,m) vector.
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *      - @c opts.TT m-by-nb matrix.
 *        In the representation of the block reflector.
 *        On exit, TT( 0:k, 0:nb ) contains blocks used to build Q :
 *        \[
 *            Q^H
 *            =
 *            [ I - W(0:nb,0:k)^T * TT(0:nb,0:nb) * conj(W(0:nb,0:k)) ]
 *            *
 *            [ I - W(nb:2nb,0:k)^T * TT(nb:2nb,0:nb) * conj(W(nb:2nb,0:k)) ]
 *            *
 *            ...
 *        \]

 *
 * @ingroup computational
 */
template <typename matrix_t, typename vector_t>
int gelqf(matrix_t& A,
          vector_t& tau,
          const gelqf_opts_t<matrix_t, size_type<matrix_t> >& opts = {})
{
    Create<matrix_t> new_matrix;

    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t nb = opts.nb;

    // check arguments
    tlapack_check((idx_t)size(tau) >= k);

    // Allocate or get workspace
    Workspace sparework;
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo;
        gelqf_worksize(A, tau, workinfo, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // TT serves two functions, it stores the triangular factors
    // in the compact WY representation and the parts that do not
    // store triangular factors yet are used as workspace.
    // If passed as an optional arguments, it is used so that the 
    // triangular factors can be reused later. Otherwise, workspace is used.
    matrix_t TT = [&]() {
        if(opts.TT != NULL){
            tlapack_check( nrows(*opts.TT) >= m and ncols(*opts.TT) >= nb );
            sparework = work;
            return *opts.TT;
        }
        return new_matrix(work, m, nb, sparework);
    }();

    // Main computational loop
    for (idx_t j = 0; j < k; j += nb) {
        idx_t ib = std::min<idx_t>(nb, k - j);

        // Compute the LQ factorization of the current block A(j:j+ib-1,j:n)
        auto TT1 = slice(TT, range(j, j + ib), range(0, ib));
        auto A11 = slice(A, range(j, j + ib), range(j, n));
        auto tauw1 = slice(tau, range( j, j+ib ));

        gelq2(A11, tauw1, sparework);

        if (j + ib < k) {
            // Form the triangular factor of the block reflector H = H(j) H(j+1)
            // . . . H(j+ib-1)
            larft(Direction::Forward, StoreV::Rowwise, A11, tauw1, TT1);

            // Apply H to A(j+ib:m,j:n) from the right
            auto A12 = slice(A, range(j + ib, m), range(j, n));

            workspace_opts_t<void> work1(
                slice(TT, range(j + ib, m), range(0, ib)));
            larfb(Side::Right, Op::NoTrans, Direction::Forward, StoreV::Rowwise,
                  A11, TT1, A12, work1);
        }
    }

    return 0;
}
}  // namespace tlapack
#endif  // TLAPACK_GELQF_HH
