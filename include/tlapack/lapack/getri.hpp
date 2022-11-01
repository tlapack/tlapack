/// @file getri.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GETRI_HH
#define TLAPACK_GETRI_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/swap.hpp"

#include "tlapack/lapack/getri_uxli.hpp"
#include "tlapack/lapack/getri_uili.hpp"

namespace tlapack {

    enum class GetriVariant : char {
        UILI = 'D', ///< Method D from doi:10.1137/1.9780898718027
        UXLI = 'C'  ///< Method C from doi:10.1137/1.9780898718027
    };

    template< class work_t >
    struct getri_opts_t
    {
        GetriVariant variant = GetriVariant::UILI;
        work_t* work_ptr = nullptr;
    };

    /** getri computes inverse of a general n-by-n matrix A
     *
     * @return = 0: successful exit
     * @return = i+1: if U(i,i) is exactly zero.  The triangular
     *          matrix is singular and its inverse can not be computed.
     *
     * @param[in,out] A n-by-n matrix.
     *      On entry, the factors L and U from the factorization P A = L U.
     *          L is stored in the lower triangle of A, the unit diagonal elements of L are not stored.
     *          U is stored in the upper triangle of A.
     *      On exit, inverse of A is overwritten on A.
     *      
     * @param[in] Piv pivot vector of size at least n.
     * 
     * @param[in] opts Options.
     *      - variant:
     *          - UILI = 'D', ///< Method D from doi:10.1137/1.9780898718027
     *          - UXLI = 'C'  ///< Method C from doi:10.1137/1.9780898718027
     *      - work: Check the correct variant to obtain details.
     *      
     * @ingroup group_solve
     */
    template< class matrix_t, class vector_t, class work_t >
    int getri( matrix_t& A, const vector_t &Piv, const getri_opts_t<work_t>& opts )
    {
        using idx_t = size_type< matrix_t >;

        // check arguments
        tlapack_check( nrows(A)==ncols(A));
        
        // Constants
        const idx_t n = ncols(A);

        // Call variant
        int info;
        if( opts.variant == GetriVariant::UXLI )
            info = getri_uxli( A, *opts.work_ptr );
        else
            info = getri_uili( A );

        // Return is matrix is not invertible
        if( info != 0 )
            return info;
        
        // swap columns of X to find A^{-1} since A^{-1}=X P
        for(idx_t j=n; j-->0;) {
            if(Piv[j]!=j){
                auto vect1=tlapack::col(A,j);
                auto vect2=tlapack::col(A,Piv[j]);
                tlapack::swap(vect1,vect2);
            }
        }
        
        return 0;
        
    } // getri

} // lapack

#endif // TLAPACK_GETRI_HH



