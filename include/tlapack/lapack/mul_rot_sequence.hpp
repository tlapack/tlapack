/// @file mul_rot_sequence.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MUL_ROT_SEQUENCE_HH
#define TLAPACK_MUL_ROT_SEQUENCE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/lapack/laset.hpp"

namespace tlapack {

/** Worspace query of mul_rot_sequence()
 *
 * @param[in] side
 *      Indicates the side of A to which the rotations will be applied
 *
 * @param[in] C real (k-1)-by-nr matrix.
 *      Contains the cosines of the rotations
 *
 * @param[in] S (k-1)-by-nr matrix.
 *      Contains the sines of the rotations
 *
 * @param[in] A matrix.
 *      If side == Side::Left, a k-by-n matrix
 *      If side == Side::Right, an m-by-k matrix
 *
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @param[in] opts Options.
 *
 * @ingroup workspace_query
 */
template <typename C_t, typename S_t, typename A_t>
inline constexpr void mul_rot_sequence_worksize(
    Side side,
    const C_t& C,
    const S_t& S,
    const A_t& A,
    workinfo_t& workinfo,
    const workspace_opts_t<>& opts = {})
{
    using idx_t = size_type<A_t>;
    using TS = type_t<S_t>;
    using TA = type_t<A_t>;

    // constants
    const idx_t nr = ncols(C);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    const idx_t t = side == Side::Left ? n : m;

    // workspace for accumulated rotation matrix
    workinfo_t myworkinfo1(sizeof(TS) * 2 * nr, 2 * nr);
    // workspace for multiplication in A
    workinfo_t myworkinfo2(sizeof(TA) * 2 * nr, t);
    myworkinfo1 += myworkinfo2;
    workinfo.minMax(myworkinfo1);
}

/** Multiply a matrix A with a sequence of rotations stored in matrices C and S.
 *
 * @param[in] side
 *      Indicates the side of A to which the rotations will be applied
 *
 * @param[in] C real (k-1)-by-nr matrix.
 *      Contains the cosines of the rotations
 *
 * @param[in] S (k-1)-by-nr matrix.
 *      Contains the sines of the rotations
 *
 * @param[in] A matrix.
 *      If side == Side::Left, a k-by-n matrix
 *      If side == Side::Right, an m-by-k matrix
 *
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @param[in] opts Options.
 *
 * @ingroup workspace_query
 */
template <typename C_t, typename S_t, typename A_t>
int mul_rot_sequence(Side side,
                     const C_t& C,
                     const S_t& S,
                     A_t& A,
                     const workspace_opts_t<>& opts = {})
{
    // Note, we create two types of matrices to support updating a complex
    // matrix with real rotations (this is used in the svd algorithm).
    Create<S_t> new_matrix_s;
    Create<A_t> new_matrix_a;

    using idx_t = size_type<A_t>;
    using TS = type_t<S_t>;
    using TA = type_t<A_t>;
    using pair = std::pair<idx_t, idx_t>;
    using real_t = real_type<TS>;

    // constants
    const idx_t k = nrows(C) + 1;
    const idx_t nr = ncols(C);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const real_t one(1);
    const real_t zero(0);

    // Allocate or get workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo;
        mul_rot_sequence_worksize(side, C, S, A, workinfo, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    Workspace sparework;
    auto Q = new_matrix_s(work, 2 * nr, 2 * nr, sparework);

    // TODO: support forward mode rotations
    // TODO: exploit initial diagonal structure of Qi when accumulating
    // rotations
    for (idx_t i = 0; i <= k; i = i + nr) {
        idx_t nb = min(2 * nr, k - i);

        //
        // Accumulate the rotations into Q
        //
        auto Qi = slice(Q, pair{0, nb}, pair{0, nb});
        laset(Uplo::General, zero, one, Qi);

        if (nb == k) {
            // Accumulate all the rotations into a single rotation matrix
            for (idx_t i2 = 0; i2 < nr; ++i2) {
                for (idx_t i1 = 0; i1 < nb - 1; ++i1) {
                    auto q1 = col(Q, i1);
                    auto q2 = col(Q, i1 + 1);
                    rot(q1, q2, C(i + i1, i2), conj(S(i + i1, i2)));
                }
            }
        }
        else if (i == 0) {
            // Treat the first block separately
            for (idx_t i2 = 0; i2 < nr; ++i2) {
                for (idx_t i1 = 0; i1 < nb - i2 - 1; ++i1) {
                    auto q1 = col(Q, i1);
                    auto q2 = col(Q, i1 + 1);
                    rot(q1, q2, C(i + i1, i2), conj(S(i + i1, i2)));
                }
            }
        }
        else if (i + nb == k) {
            // Treat the last block separately
            for (idx_t i2 = 0; i2 < nr; ++i2) {
                for (idx_t i1 = nr - 1 - i2; i1 < nb - 1; ++i1) {
                    auto q1 = col(Q, i1);
                    auto q2 = col(Q, i1 + 1);
                    rot(q1, q2, C(i + i1, i2), conj(S(i + i1, i2)));
                }
            }
        }
        else {
            for (idx_t i2 = 0; i2 < nr; ++i2) {
                for (idx_t i1 = nr - 1 - i2; i1 < nb - i2 - 1; ++i1) {
                    auto q1 = col(Q, i1);
                    auto q2 = col(Q, i1 + 1);
                    rot(q1, q2, C(i + i1, i2), conj(S(i + i1, i2)));
                }
            }
        }

        //
        // Apply Q to A
        //
        if (side == Side::Left) {
            auto B = new_matrix_a(sparework, nb, n);
            auto Ai = rows(A, pair{i, i + nb});
            gemm(Op::ConjTrans, Op::NoTrans, one, Qi, Ai, B);
            lacpy(Uplo::General, B, Ai);
        }
        else {
            auto B = new_matrix_a(sparework, m, nb);
            auto Ai = cols(A, pair{i, i + nb});
            gemm(Op::NoTrans, Op::NoTrans, one, Ai, Qi, B);
            lacpy(Uplo::General, B, Ai);
        }

        // If the last block has been handled, break
        if (nb == k or i + nb == k) break;
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_MUL_ROT_SEQUENCE_HH
