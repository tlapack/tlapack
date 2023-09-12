/// @file hessenberg_rq.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HESSENBERG_RQ_HH
#define TLAPACK_HESSENBERG_RQ_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"

namespace tlapack {

/** Calculates the RQ factorization of a Hessenberg matrix using a sequence of
 * Givens rotations
 *
 * @return  0 if success
 *
 * @param[in,out] H n-by-n Hessenberg matrix.
 *
 * @param[in] c Real vector of length n-1.
 *     Cosines of the rotations
 *
 * @param[in] s Vector of length n-1.
 *     Sines of the rotations
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX H_t, TLAPACK_SVECTOR C_t, TLAPACK_SVECTOR S_t>
int hessenberg_rq(H_t& H, C_t& c, S_t& s)
{
    using T = type_t<H_t>;
    using idx_t = size_type<H_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(H);
    const idx_t k = n - 1;

    // quick return
    if (k < 1) return 0;

    for (idx_t i = n - 1; i > 0; --i) {
        // Generate rotation to reduce
        rotg(H(i, i), H(i, i - 1), c[i - 1], s[i - 1]);
        s[i - 1] = -s[i - 1];
        H(i, i - 1) = (T)0;

        // Apply rotation to H
        auto h1 = slice(H, range(0, i), i - 1);
        auto h2 = slice(H, range(0, i), i);
        rot(h1, h2, c[i - 1], conj(s[i - 1]));
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_HESSENBERG_RQ_HH
