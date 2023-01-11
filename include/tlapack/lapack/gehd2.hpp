/// @file gehd2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dgehd2.f
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEHD2_HH
#define TLAPACK_GEHD2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack {

/** Worspace query of gehd2()
 * 
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      It is assumed that A is already upper Hessenberg in columns
 *      0:ilo and rows ihi:n and is already upper triangular in
 *      columns ihi+1:n and rows 0:ilo.
 *      0 <= ilo <= ihi <= max(1,n).
 * @param[in] A n-by-n matrix.
 *      On entry, the n by n general matrix to be reduced.
 * @param tau Not referenced.
 *
 * @param[in] opts Options.
 * 
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template< class matrix_t, class vector_t >
inline constexpr
void gehd2_worksize(
    size_type< matrix_t > ilo, size_type< matrix_t > ihi, const matrix_t& A,
    const vector_t &tau, workinfo_t& workinfo,
    const workspace_opts_t<>& opts = {} )
{
    using idx_t = size_type< matrix_t >;
    using pair  = pair<idx_t,idx_t>;

    // constants
    const idx_t n = ncols(A);

    if( ilo+1 < ihi ) {
        const auto v = slice( A, pair{ilo+1,ihi}, ilo );
        
        auto C0 = slice( A, pair{0,ihi}, pair{ilo+1,ihi} );
        larf_worksize( right_side, forward, v, tau[0], C0, workinfo, opts );
        
        auto C1 = slice( A, pair{ilo+1,ihi}, pair{ilo+1,n} );
        larf_worksize( left_side, forward, v, tau[0], C1, workinfo, opts );
    }
}

/** Reduces a general square matrix to upper Hessenberg form
 * 
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_ilo H_ilo+1 ... H_ihi,
 * \]
 * Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i] = 0; v[i+1] = 1,
 * \]
 * with v[i+2] through v[ihi] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 * 
 * @return  0 if success
 * 
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      It is assumed that A is already upper Hessenberg in columns
 *      0:ilo and rows ihi:n and is already upper triangular in
 *      columns ihi+1:n and rows 0:ilo.
 *      0 <= ilo <= ihi <= max(1,n).
 * @param[in,out] A n-by-n matrix.
 *      On entry, the n by n general matrix to be reduced.
 *      On exit, the upper triangle and the first subdiagonal of A
 *      are overwritten with the upper Hessenberg matrix H, and the
 *      elements below the first subdiagonal, with the array TAU,
 *      represent the orthogonal matrix Q as a product of elementary
 *      reflectors. See Further Details.
 * @param[out] tau Real vector of length n-1.
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 * 
 * @ingroup computational
 */
template< class matrix_t, class vector_t >
int gehd2( size_type< matrix_t > ilo, size_type< matrix_t > ihi, matrix_t& A, vector_t &tau, const workspace_opts_t<>& opts = {} )
{
    using idx_t = size_type< matrix_t >;
    using pair  = pair<idx_t,idx_t>;

    // constants
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check_false( ncols(A) != nrows(A) );
    tlapack_check_false( (idx_t) size(tau)  < n-1 );

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]()
    {
        workinfo_t workinfo;
        gehd2_worksize( ilo, ihi, A, tau, workinfo, opts );
        return alloc_workspace( localworkdata, workinfo, opts.work );
    }();
    
    // Options to forward
    auto&& larfOpts = workspace_opts_t<>{ work };

    for(idx_t i = ilo; i < ihi-1; ++i) {

        // Define x := A[i+1:ihi,i]
        auto v = slice(A, pair{i+1,ihi}, i);

        // Generate the (i+1)-th elementary Householder reflection on x
        larfg( v, tau[i] );

        // Apply Householder reflection from the right to A[0:ihi,i+1:ihi]
        auto C0 = slice( A, pair{0,ihi}, pair{i+1,ihi} );
        larf( right_side, forward, v, tau[i], C0, larfOpts );

        // Apply Householder reflection from the left to A[i+1:ihi,i+1:n-1]
        auto C1 = slice( A, pair{i+1,ihi}, pair{i+1,n} );
        larf( left_side, forward, v, conj(tau[i]), C1, larfOpts );
	}

    return 0;
}

} // lapack

#endif // TLAPACK_GEHD2_HH
