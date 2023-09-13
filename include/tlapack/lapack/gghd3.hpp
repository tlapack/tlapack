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
 * @param[in] opts Options.
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
    tlapack_check(ilo >= 0 && ilo < n);
    tlapack_check(ihi > ilo && ihi <= n);
    tlapack_check(n == nrows(A));
    tlapack_check(n == ncols(B));
    tlapack_check(n == nrows(B));
    tlapack_check(n == ncols(Q));
    tlapack_check(n == nrows(Q));
    tlapack_check(n == ncols(Z));
    tlapack_check(n == nrows(Z));

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

    std::vector<T> Qt_;
    auto Qt = new_matrix(Qt_, 2 * nb, 2 * nb);
    std::vector<T> C_;
    auto C = new_matrix(C_, 2 * nb, n);
    auto D = new_matrix(C_, n, 2 * nb);

    for (idx_t j = ilo; j + 2 < ihi; j = j + nb) {
        // Number of columns to be reduced
        idx_t nnb = std::min<idx_t>(nb, ihi - 2 - j);
        // Number of 2*nnb x 2*nnb orthogonal factors
        idx_t n2nb = (ihi - j - 2) / nnb - 1;
        // Size of the last orthogonal factor
        idx_t nblst = ihi - j - 1 - n2nb * nnb;

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
        // Accumulate the left rotations into unitary matrices and use those to
        // apply the rotations efficiently.
        //
        {
            //
            // Last block is treated separately
            //
            auto Qt2 = slice(Qt, range(0, nblst), range(0, nblst));
            laset(GENERAL, (T)0, (T)1, Qt2);

            for (idx_t jb = 0; jb < nnb; ++jb) {
                for (idx_t i = nblst - 1; i > jb; --i) {
                    auto q1 = slice(Qt2, range(i - 1 - jb, nblst), i - 1);
                    auto q2 = slice(Qt2, range(i - 1 - jb, nblst), i);
                    rot(q1, q2, Cl(j - ilo + nnb * n2nb + i - 1, jb),
                        conj(Sl(j - ilo + nnb * n2nb + i - 1, jb)));
                }
            }

            auto A2 = slice(A, range(ihi - nblst, ihi), range(j + nnb, n));
            auto C2 = slice(C, range(0, nblst), range(j + nnb, n));
            gemm(CONJ_TRANS, NO_TRANS, (T)1, Qt2, A2, C2);
            lacpy(GENERAL, C2, A2);

            if (ihi < n) {
                auto B2 = slice(B, range(ihi - nblst, ihi), range(ihi, n));
                auto C3 = slice(C, range(0, nblst), range(ihi, n));
                gemm(CONJ_TRANS, NO_TRANS, (T)1, Qt2, B2, C3);
                lacpy(GENERAL, C3, B2);
            }

            auto Q2 = cols(Q, range(ihi - nblst, ihi));
            auto D2 = cols(D, range(0, nblst));
            gemm(NO_TRANS, NO_TRANS, (T)1, Q2, Qt2, D2);
            lacpy(GENERAL, D2, Q2);
        }
        for (idx_t ib = n2nb - 1; ib != (idx_t)-1; ib--) {
            auto Qt2 = slice(Qt, range(0, 2 * nnb), range(0, 2 * nnb));
            laset(GENERAL, (T)0, (T)1, Qt2);
            for (idx_t jb = 0; jb < nnb; ++jb) {
                for (idx_t i = nnb + jb; i > jb; --i) {
                    auto q1 =
                        slice(Qt2, range(i - 1 - jb, nnb + jb + 1), i - 1);
                    auto q2 = slice(Qt2, range(i - 1 - jb, nnb + jb + 1), i);
                    rot(q1, q2, Cl(j - ilo + ib * nnb + i - 1, jb),
                        conj(Sl(j - ilo + ib * nnb + i - 1, jb)));
                }
            }

            auto A2 =
                slice(A, range(j + 1 + nnb * ib, j + 1 + nnb * ib + 2 * nnb),
                      range(j + nnb, n));
            auto C2 = slice(C, range(0, 2 * nnb), range(j + nnb, n));
            gemm(CONJ_TRANS, NO_TRANS, (T)1, Qt2, A2, C2);
            lacpy(GENERAL, C2, A2);

            if (ihi < n) {
                auto B2 = slice(
                    B, range(j + 1 + nnb * ib, j + 1 + nnb * ib + 2 * nnb),
                    range(ihi, n));
                auto C3 = slice(C, range(0, 2 * nnb), range(ihi, n));
                gemm(CONJ_TRANS, NO_TRANS, (T)1, Qt2, B2, C3);
                lacpy(GENERAL, C3, B2);
            }

            auto Q2 =
                cols(Q, range(j + 1 + nnb * ib, j + 1 + nnb * ib + 2 * nnb));
            auto D2 = cols(D, range(0, 2 * nnb));
            gemm(NO_TRANS, NO_TRANS, (T)1, Q2, Qt2, D2);
            lacpy(GENERAL, D2, Q2);
        }

        //
        // Accumulate the right rotations into unitary matrices and use those to
        // apply the rotations efficiently.
        //
        {
            // Last block is treated separately
            auto Qt2 = slice(Qt, range(0, nblst), range(0, nblst));
            laset(GENERAL, (T)0, (T)1, Qt2);

            for (idx_t jb = 0; jb < nnb; ++jb) {
                for (idx_t i = nblst - 1; i > jb; --i) {
                    auto q1 = slice(Qt2, range(i - 1 - jb, nblst), i - 1);
                    auto q2 = slice(Qt2, range(i - 1 - jb, nblst), i);
                    rot(q1, q2, Cr(j - ilo + nnb * n2nb + i - 1, jb),
                        conj(Sr(j - ilo + nnb * n2nb + i - 1, jb)));
                }
            }

            if (j > 0) {
                auto A2 = slice(A, range(0, j), range(ihi - nblst, ihi));
                auto D2 = slice(D, range(0, j), range(0, nblst));
                gemm(NO_TRANS, NO_TRANS, (T)1, A2, Qt2, D2);
                lacpy(GENERAL, D2, A2);

                auto B2 = slice(B, range(0, j), range(ihi - nblst, ihi));
                gemm(NO_TRANS, NO_TRANS, (T)1, B2, Qt2, D2);
                lacpy(GENERAL, D2, B2);
            }

            auto Z2 = cols(Z, range(ihi - nblst, ihi));
            auto D2 = cols(D, range(0, nblst));
            gemm(NO_TRANS, NO_TRANS, (T)1, Z2, Qt2, D2);
            lacpy(GENERAL, D2, Z2);
        }
        for (idx_t ib = n2nb - 1; ib != (idx_t)-1; ib--) {
            auto Qt2 = slice(Qt, range(0, 2 * nnb), range(0, 2 * nnb));
            laset(GENERAL, (T)0, (T)1, Qt2);
            for (idx_t jb = 0; jb < nnb; ++jb) {
                for (idx_t i = nnb + jb; i > jb; --i) {
                    auto q1 =
                        slice(Qt2, range(i - 1 - jb, nnb + jb + 1), i - 1);
                    auto q2 = slice(Qt2, range(i - 1 - jb, nnb + jb + 1), i);
                    rot(q1, q2, Cr(j - ilo + ib * nnb + i - 1, jb),
                        conj(Sr(j - ilo + ib * nnb + i - 1, jb)));
                }
            }

            if (j > 0) {
                auto A2 =
                    slice(A, range(0, j),
                          range(j + 1 + nnb * ib, j + 1 + nnb * ib + 2 * nnb));
                auto D2 = slice(D, range(0, j), range(0, 2 * nnb));
                gemm(NO_TRANS, NO_TRANS, (T)1, A2, Qt2, D2);
                lacpy(GENERAL, D2, A2);

                auto B2 =
                    slice(B, range(0, j),
                          range(j + 1 + nnb * ib, j + 1 + nnb * ib + 2 * nnb));
                gemm(NO_TRANS, NO_TRANS, (T)1, B2, Qt2, D2);
                lacpy(GENERAL, D2, B2);
            }

            auto Z2 =
                cols(Z, range(j + 1 + nnb * ib, j + 1 + nnb * ib + 2 * nnb));
            auto D2 = cols(D, range(0, 2 * nnb));
            gemm(NO_TRANS, NO_TRANS, (T)1, Z2, Qt2, D2);
            lacpy(GENERAL, D2, Z2);
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GGHD3_HH
