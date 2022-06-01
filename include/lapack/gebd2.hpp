/// @file gebd2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/blob/master/SRC/cgebd2.f
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

namespace tlapack {

/** 
* hand-crafted  reduction to bidiagonal form of a general m-by-n matrix A, level 2, right looking algorithm
* @param[in,out] A
*/
template <typename matrix_t, class vector_t>
int gebd2( matrix_t & A, vector_t & tauv, vector_t & tauw, vector_t & work ){

    using idx_t = size_type< matrix_t >;
    using T = type_t< matrix_t >;
    using real_t = real_type<T>;
    using range = std::pair<idx_t, idx_t>;


    const idx_t m = nrows( A );
    const idx_t n = ncols( A );

    for (idx_t j = 0; j < n; ++j) {

        auto v = slice(A, range(j, m), j);
        
        larfg(v, tauv[j]); //generate the vertical reflector v

        if( j < n-1 ){ 
            auto A11 = slice(A, range(j, m), range(j+1, n));
            larf(Side::Left, v, conj(tauv[j]), A11, work);
        }

        if( j < n-1 ){
            auto w = slice(A, j, range(j+1, n)); 
            for (idx_t i = 0; i < n-j-1; ++i)
                w[i] = conj(w[i]); // see LAPACK cgebd2

            larfg(w, tauw[j]); // generate the horizontal reflector w

            if( j < m-1 ){ 
                auto B11 = slice(A, range(j+1, m), range(j+1, n));
                larf(Side::Right, w, tauw[j], B11, work);
            }
            // for (idx_t i = 0; i < n-j-1; ++i) 
            //     w[i] = conj(w[i]); 
                // this "conjugate back" step is originally from LAPACK
                // however, it's likely that we don't need it 
                // (along with no conjugation in the "if (bidg == 1)" in ungl2 ).
        }
    }

    return 0;
}
}
#endif //GEBD2