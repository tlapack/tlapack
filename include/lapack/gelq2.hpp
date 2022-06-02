/// @file gelq2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgelq2.f
//
// Copyright (c) 2014-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_GELQ2_HH__
#define __TLAPACK_GELQ2_HH__

#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace tlapack {

/** 
* hand-crafted LQ factorization for a general m-by-n row major matrix A (m < n), level 2, right looking algorithm
* outputing A in terms of L and reflector w's
* @param[in,out] A
*/

template <typename matrix_t, class vector_t, class work_t>
int gelq2(matrix_t & A, vector_t & tauw, work_t & work ){
    using idx_t = size_type< matrix_t >;
    using range = std::pair<idx_t, idx_t>;


    const idx_t m = nrows( A );
    const idx_t n = ncols( A );

    const idx_t k = std::min(m, n);

    for (idx_t j = 0; j < k; ++j) {
        auto w = slice(A, j, range(j, n)); 
        //for loop to conj w.
        for (idx_t i = 0; i < n-j; ++i)
            w[i] = conj(w[i]);

        larfg(w, tauw[j]); //gen the horizontal reflector w's

        if( j < k-1 || k < m){ 
            auto Q11 = slice(A, range(j+1, m), range(j, n));
            larf(Side::Right, w, tauw[j], Q11, work);
        }
        for (idx_t i = 0; i < n-j; ++i) //for loop to conj w back.
            w[i] = conj(w[i]);
    }

    return 0;
}
}


#endif //GELQ2