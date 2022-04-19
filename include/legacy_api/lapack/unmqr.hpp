/// @file unmqr.hpp Multiplies the general m-by-n matrix C by Q from lapack::geqrf()
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_UNMQR_HH__
#define __TLAPACK_LEGACY_UNMQR_HH__

#include <memory>
#include "lapack/unmqr.hpp"

namespace lapack {

/** Multiplies the general m-by-n matrix C by Q from lapack::geqrf() using a blocked code as follows:
 *
 * @param[in] side
 *     - lapack::Side::Left:  apply $Q$ or $Q^H$ from the Left;
 *     - lapack::Side::Right: apply $Q$ or $Q^H$ from the Right.
 *
 * @param[in] trans
 *     - lapack::Op::NoTrans:   No transpose, apply $Q$;
 *     - lapack::Op::ConjTrans: Conjugate transpose, apply $Q^H$.
 *
 * @param[in] m
 *     The number of rows of the matrix C. m >= 0.
 *
 * @param[in] n
 *     The number of columns of the matrix C. n >= 0.
 *
 * @param[in] k
 *     The number of elementary reflectors whose product defines
 *     the matrix Q.
 *     - If side = Left,  m >= k >= 0;
 *     - if side = Right, n >= k >= 0.
 *
 * @param[in] A
 *     - If side = Left,  the m-by-k matrix A, stored in an lda-by-k array;
 *     - if side = Right, the n-by-k matrix A, stored in an lda-by-k array.
 *     \n
 *     The i-th column must contain the vector which defines the
 *     elementary reflector H(i), for i = 1, 2, ..., k, as returned by
 *     lapack::geqrf() in the first k columns of its array argument A.
 *
 * @param[in] lda
 *     The leading dimension of the array A.
 *     - If side = Left,  lda >= max(1,m);
 *     - if side = Right, lda >= max(1,n).
 *
 * @param[in] tau
 *     The vector tau of length k.
 *     tau[i] must contain the scalar factor of the elementary
 *     reflector H(i), as returned by lapack::geqrf().
 *
 * @param[in,out] C
 *     The m-by-n matrix C, stored in an ldc-by-n array.
 *     On entry, the m-by-n matrix C.
 *     On exit, C is overwritten by
 *     $Q C$ or $Q^H C$ or $C Q^H$ or $C Q$.
 *
 * @param[in] ldc
 *     The leading dimension of the array C. ldc >= max(1,m).
 * 
 * @see unmqr(
    side_t side, trans_t trans,
    const matrixA_t& A, const tau_t& tau,
    matrixC_t& C, opts_t&& opts )
 *
 * @ingroup geqrf
 */
template< class side_t, class trans_t, typename TA, typename TC >
inline int unmqr(
    side_t side, trans_t trans,
    blas::idx_t m, blas::idx_t n, blas::idx_t k,
    TA const* A, blas::idx_t lda,
    TA const* tau,
    TC* C, blas::idx_t ldc )
{
    typedef blas::scalar_type<TA,TC> scalar_t;
    using blas::internal::colmajor_matrix;
    using blas::internal::vector;

    // Constants
    const idx_t nb = 32; // number of blocks
                         /// TODO: Improve me!
    const idx_t nw = (side == Side::Left)
                ? ( (n >= 1) ? n : 1 )
                : ( (m >= 1) ? m : 1 );

    // check arguments
    lapack_error_if( side != Side::Left &&
                     side != Side::Right, -1 );
    lapack_error_if( trans != Op::NoTrans &&
                     trans != Op::Trans &&
                     trans != Op::ConjTrans, -2 );
    
    // Allocate work
    std::unique_ptr<scalar_t[]> _work( new scalar_t[ nb * (nw + nb) ] );
                
    // Matrix views
    const auto _A = (side == Side::Left)
            ? colmajor_matrix<TA>( (TA*)A, m, k, lda )
            : colmajor_matrix<TA>( (TA*)A, n, k, lda );
    const auto _tau = vector( (TA*)tau, k );
    auto _C = colmajor_matrix<TC>( C, m, n, ldc );
    auto _W = colmajor_matrix<scalar_t>( &_work[0], nb, nw+nb );

    // Options
    struct {
        idx_t nb;
        decltype(_W)* workPtr;
    } opts = { nb, &_W };
    
    return unmqr( side, trans, _A, _tau, _C, std::move(opts) );
}

}

#endif // __TLAPACK_LEGACY_UNMQR_HH__
