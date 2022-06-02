/// @file gelqf.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgelqf.f
//
// Copyright (c) 2014-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_GELQF_HH__
#define __TLAPACK_GELQF_HH__

#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace tlapack {

/**
 * hand-crafted general LQ factorization, level 3 BLAS, right looking algorithm
 * @param[in,out] A
 */

template <typename matrix_t, class vector_t, class work_t>
int gelqf(matrix_t &A, matrix_t &TT, vector_t &tauw, work_t &work, const size_type<matrix_t> &nb)
{

    // type alias for indexes
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t j = 0; j < m; j += nb)
    {
        idx_t ib = std::min<idx_t>(nb, m - j);

        auto TT1 = slice(TT, range(j, j + ib), range(0, ib));
        auto A11 = slice(A, range(j, j + ib), range(j, n));

        auto tauw1 = slice(tauw, range(j, j + ib));

        gelq2(A11, tauw1, work);

        larft(Direction::Forward, 
            StoreV::Rowwise, A11, tauw1, TT1);

        if( j+ib < m ){                 
            auto A12 = slice(A, range(j+ib, m), range(j, n));
            auto work1 = slice(TT, range(j+ib, m), range(0, ib));

            larfb(
                    Side::Right, 
                    Op::NoTrans,
                    Direction::Forward, 
                    StoreV::Rowwise, 
                    A11, TT1, A12, work1);
        }
    }

    return 0;
}
}
#endif // GELQF