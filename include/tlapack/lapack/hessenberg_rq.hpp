/// @file hessenberg_rq.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
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

/** Applies a sequence of rotations to an upper triangular matrix T
 * from the left (making it an upper Hessenberg matrix) and reduces
 * that matrix back to upper triangular form using rotations from the right.
 *
 * i.e. rot_sequence(LEFT_SIDE, FORWARD, cl, sl, T1) = rot_sequence(RIGHT_SIDE,
 * FORWARD, cr, sr, T2)
 *
 * This is an important component of gghd3
 *
 * @param[in,out] T n-by-n Upper triangular matrix.
 *
 * @param[in] cl Real vector of length n-1.
 *     Cosines of the left rotations
 *
 * @param[in] sl Vector of length n-1.
 *     Sines of the left rotations
 *
 * @param[out] cr Real vector of length n-1.
 *     Cosines of the right rotations
 *
 * @param[out] sr Vector of length n-1.
 *     Sines of the right rotations
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX T_t,
          TLAPACK_SVECTOR CL_t,
          TLAPACK_SVECTOR SL_t,
          TLAPACK_SVECTOR CR_t,
          TLAPACK_SVECTOR SR_t>
inline void hessenberg_rq(T_t& T, CL_t& cl, SL_t& sl, CR_t& cr, SR_t& sr)
{
    using TA = type_t<T_t>;
    using idx_t = size_type<T_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(T);
    const idx_t k = n - 1;

    // quick return
    if (k < 1) return;

    // Apply rotations to last column
    for (idx_t j = n - 2; j != (idx_t)-1; --j) {
        TA temp = cl[j] * T(j, n - 1) + sl[j] * T(j + 1, n - 1);
        T(j + 1, n - 1) = -conj(sl[j]) * T(j, n - 1) + cl[j] * T(j + 1, n - 1);
        T(j, n - 1) = temp;
    }

    for (idx_t i = n - 1; i > 0; --i) {
        // Apply rotations from the left to the next column
        for (idx_t j = i - 1; j != (idx_t)-1; --j) {
            TA temp = cl[j] * T(j, i - 1) + sl[j] * T(j + 1, i - 1);
            T(j + 1, i - 1) =
                -conj(sl[j]) * T(j, i - 1) + cl[j] * T(j + 1, i - 1);
            T(j, i - 1) = temp;
        }

        // Generate rotation to reduce T(i, i-1)
        rotg(T(i, i), T(i, i - 1), cr[i - 1], sr[i - 1]);
        sr[i - 1] = -sr[i - 1];
        T(i, i - 1) = (TA)0;

        // Apply rotation from the right
        auto t1 = slice(T, range(0, i), i - 1);
        auto t2 = slice(T, range(0, i), i);
        rot(t1, t2, cr[i - 1], conj(sr[i - 1]));
    }
}

}  // namespace tlapack

#endif  // TLAPACK_HESSENBERG_RQ_HH
