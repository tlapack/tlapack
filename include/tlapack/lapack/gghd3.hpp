/// @file gghd3.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zgghd3.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GGHD3_HH
#define TLAPACK_GGHD3_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"
#include "tlapack/lapack/hessenberg_rq.hpp"
#include "tlapack/lapack/rot_sequence.hpp"

namespace tlapack {

/**
 * Options struct for gghd3
 */
struct Gghd3Opts {
    size_t nb = 32;  ///< Block size
};

/** Reduces a pair of real square matrices (A, B) to generalized upper
 *  Hessenberg form using unitary transformations, where A is a general matrix
 *  and B is upper triangular.
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
 * @param[in] wantq boolean
 * @param[in] wantz boolean
 * @param[in] ilo integer
 * @param[in] ihi integer
 * @param[in,out] A n-by-n matrix.
 * @param[in,out] B n-by-n matrix.
 * @param[in,out] Q n-by-n matrix.
 * @param[in,out] Z n-by-n matrix.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX A_t,
          TLAPACK_SMATRIX B_t,
          TLAPACK_SMATRIX Q_t,
          TLAPACK_SMATRIX Z_t>
int gghd3(bool wantq,
          bool wantz,
          size_type<A_t> ilo,
          size_type<A_t> ihi,
          A_t& A,
          B_t& B,
          Q_t& Q,
          Z_t& Z,
          const Gghd3Opts& opts = {})
{
    using idx_t = size_type<A_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<A_t>;
    using real_t = real_type<T>;
    using r_matrix_t = real_type<A_t>;

    Create<A_t> new_matrix;
    Create<r_matrix_t> new_real_matrix;

    // constants
    const idx_t n = ncols(A);
    const idx_t nb = opts.nb;
    const idx_t nh = ihi - ilo - 1;

    // check arguments
    tlapack_check_false(ncols(A) != nrows(A));

    // Zero out lower triangle of B
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            B(i, j) = (T)0;

    // quick return
    if (nh <= 1) return 0;

    // Locally allocate workspace for now
    std::vector<real_t> Cl_;
    auto Cl = new_real_matrix(Cl_, nh - 1, nb);
    std::vector<T> Sl_;
    auto Sl = new_matrix(Sl_, nh - 1, nb);
    std::vector<real_t> Cr_;
    auto Cr = new_real_matrix(Cr_, nh - 1, nb);
    std::vector<T> Sr_;
    auto Sr = new_matrix(Sr_, nh - 1, nb);

    for (idx_t j = ilo; j + 2 < ihi; j = j + nb) {
        idx_t nnb = std::min<idx_t>(nb, ihi - 2 - j);

        //
        // Reduce panel j:j+nb
        //
        for (idx_t jb = 0; jb < nnb; ++jb) {
            // Update jb-th column of the block
            for (idx_t jbb = 0; jbb < jb; ++jbb) {
                for (idx_t i = ihi - 1; i > j + jbb + 1; --i) {
                    real_t c = Cl(i - ilo - 2, jbb);
                    T s = Sl(i - ilo - 2, jbb);
                    T temp = c * A(i - 1, j + jb) + s * A(i, j + jb);
                    A(i, j + jb) =
                        -conj(s) * A(i - 1, j + jb) + c * A(i, j + jb);
                    A(i - 1, j + jb) = temp;
                }
            }
            // Reduce column in A
            for (idx_t i = ihi - 1; i > j + jb + 1; --i) {
                rotg(A(i - 1, j + jb), A(i, j + jb), Cl(i - ilo - 2, jb),
                     Sl(i - ilo - 2, jb));
                A(i, j + jb) = (T)0;
            }
            auto clv = slice(Cl, range(j - ilo + jb, ihi - ilo - 2), jb);
            auto slv = slice(Sl, range(j - ilo + jb, ihi - ilo - 2), jb);
            // Apply rotations to B
            auto B2 = slice(B, range(j + jb + 1, ihi), range(j + jb + 1, ihi));
            rot_sequence(LEFT_SIDE, FORWARD, clv, slv, B2);

            // Remove fill-in from B
            auto crv = slice(Cr, range(j - ilo + jb, ihi - ilo - 2), jb);
            auto srv = slice(Sr, range(j - ilo + jb, ihi - ilo - 2), jb);
            hessenberg_rq(B2, crv, srv);
            auto B3 = slice(B, range(j, j + jb + 1), range(j + jb + 1, ihi));
            rot_sequence(RIGHT_SIDE, FORWARD, crv, srv, B3);
            // Apply rotations to A
            auto A2 = slice(A, range(j, ihi), range(j + jb + 1, ihi));
            rot_sequence(RIGHT_SIDE, FORWARD, crv, srv, A2);
        }

        //
        // This loop applies the rotations to the rest of the matrices
        // This should be optimized using BLAS
        //
        for (idx_t jb = 0; jb < nnb; ++jb) {
            auto clv = slice(Cl, range(j - ilo + jb, ihi - ilo - 2), jb);
            auto slv = slice(Sl, range(j - ilo + jb, ihi - ilo - 2), jb);
            auto crv = slice(Cr, range(j - ilo + jb, ihi - ilo - 2), jb);
            auto srv = slice(Sr, range(j - ilo + jb, ihi - ilo - 2), jb);

            auto A2 = slice(A, range(j + jb + 1, ihi), range(j + nnb, n));
            rot_sequence(LEFT_SIDE, FORWARD, clv, slv, A2);

            if (ihi < n) {
                auto B4 = slice(B, range(j + jb + 1, ihi), range(ihi, n));
                rot_sequence(LEFT_SIDE, FORWARD, clv, slv, B4);
            }
            if (j > 0) {
                auto A4 = slice(A, range(0, j), range(j + jb + 1, ihi));
                rot_sequence(RIGHT_SIDE, FORWARD, crv, srv, A4);
                auto B4 = slice(B, range(0, j), range(j + jb + 1, ihi));
                rot_sequence(RIGHT_SIDE, FORWARD, crv, srv, B4);
            }
            // Apply rotations to Q
            auto Q2 = cols(Q, range(j + jb + 1, ihi));
            rot_sequence(RIGHT_SIDE, FORWARD, clv, slv, Q2);
            // Apply rotations to Z
            auto Z2 = cols(Z, range(j + jb + 1, ihi));
            rot_sequence(RIGHT_SIDE, FORWARD, crv, srv, Z2);
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GGHD3_HH
