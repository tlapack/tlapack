/// @file lasy2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlasy2.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LASY2_HH
#define TLAPACK_LASY2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/swap.hpp"

namespace tlapack {

/** lasy2 solves the Sylvester matrix equation where the matrices are of order 1
 * or 2.
 *
 *  lasy2 solves for the N1 by N2 matrix X, 1 <= N1,N2 <= 2, in
 *  op(TL)*X + ISGN*X*op(TR) = SCALE*B,
 *
 *  where TL is N1 by N1, TR is N2 by N2, B is N1 by N2, and ISGN = 1 or
 *  -1.  op(T) = T or T**T, where T**T denotes the transpose of T.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrix_t,
          enable_if_t<is_real<type_t<matrix_t> >, bool> = true>
int lasy2(Op trans_l,
          Op trans_r,
          int isign,
          const matrix_t& TL,
          const matrix_t& TR,
          const matrix_t& B,
          type_t<matrix_t>& scale,
          matrix_t& X,
          type_t<matrix_t>& xnorm)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;

    // Functor for creating new matrices of type matrix_t
    Create<matrix_t> new_matrix;

    const idx_t n1 = ncols(TL);
    const idx_t n2 = ncols(TR);
    const T eps = ulp<T>();
    const T small_num = safe_min<T>() / eps;

    const T zero(0);
    const T one(1);
    const T eight(8);

    tlapack_check(isign == -1 or isign == 1);

    // Quick return
    if (n1 == 0 or n2 == 0) return 0;

    T sgn(isign);
    int info = 0;

    if (n1 == 1 and n2 == 1) {
        T tau1 = TL(0, 0) + sgn * TR(0, 0);
        T bet = abs(tau1);
        if (bet < small_num) {
            tau1 = small_num;
            bet = small_num;
            info = 1;
        }
        scale = one;
        const T gam = abs(B(0, 0));
        if (small_num * gam > bet) scale = one / gam;
        X(0, 0) = (B(0, 0) * scale) / tau1;
        xnorm = abs(X(0, 0));

        return info;
    }
    if ((n1 == 2 and n2 == 1) or (n1 == 1 and n2 == 2)) {
        // TODO
        return -1;
    }
    if (n1 == 2 and n2 == 2) {
        // 2x2 blocks, build a 4x4 matrix
        std::vector<T> btmp(4);
        std::vector<T> tmp(4);
        std::vector<T> T16_;
        auto T16 = new_matrix(T16_, 4, 4);
        std::vector<idx_t> jpiv(4);

        T smin = max(max(abs(TR(0, 0)), abs(TR(0, 1))),
                     max(abs(TR(1, 0)), abs(TR(1, 1))));
        smin = max(smin, max(max(abs(TL(0, 0)), abs(TL(0, 1))),
                             max(abs(TL(1, 0)), abs(TL(1, 1)))));
        smin = max(eps * smin, small_num);

        for (idx_t i = 0; i < 4; ++i)
            for (idx_t j = 0; j < 4; ++j)
                T16(i, j) = zero;

        T16(0, 0) = TL(0, 0) + sgn * TR(0, 0);
        T16(1, 1) = TL(1, 1) + sgn * TR(0, 0);
        T16(2, 2) = TL(0, 0) + sgn * TR(1, 1);
        T16(3, 3) = TL(1, 1) + sgn * TR(1, 1);

        if (trans_l == Op::Trans) {
            T16(0, 1) = TL(1, 0);
            T16(1, 0) = TL(0, 1);
            T16(2, 3) = TL(1, 0);
            T16(3, 2) = TL(0, 1);
        }
        else {
            T16(0, 1) = TL(0, 1);
            T16(1, 0) = TL(1, 0);
            T16(2, 3) = TL(0, 1);
            T16(3, 2) = TL(1, 0);
        }
        if (trans_r == Op::Trans) {
            T16(0, 2) = sgn * TR(0, 1);
            T16(1, 3) = sgn * TR(0, 1);
            T16(2, 0) = sgn * TR(1, 0);
            T16(3, 1) = sgn * TR(1, 0);
        }
        else {
            T16(0, 2) = sgn * TR(1, 0);
            T16(1, 3) = sgn * TR(1, 0);
            T16(2, 0) = sgn * TR(0, 1);
            T16(3, 1) = sgn * TR(0, 1);
        }
        btmp[0] = B(0, 0);
        btmp[1] = B(1, 0);
        btmp[2] = B(0, 1);
        btmp[3] = B(1, 1);

        // Perform elimination with pivoting to solve 4x4 system
        idx_t ipsv, jpsv;
        for (idx_t i = 0; i < 3; ++i) {
            ipsv = i;
            jpsv = i;
            // Do pivoting to get largest pivot element
            T xmax = zero;
            for (idx_t ip = i; ip < 4; ++ip) {
                for (idx_t jp = i; jp < 4; ++jp) {
                    if (abs(T16(ip, jp)) >= xmax) {
                        xmax = abs(T16(ip, jp));
                        ipsv = ip;
                        jpsv = jp;
                    }
                }
            }
            if (ipsv != i) {
                auto row1 = row(T16, ipsv);
                auto row2 = row(T16, i);
                tlapack::swap(row1, row2);
                const T temp = btmp[i];
                btmp[i] = btmp[ipsv];
                btmp[ipsv] = temp;
            }
            if (jpsv != i) {
                auto col1 = col(T16, jpsv);
                auto col2 = col(T16, i);
                tlapack::swap(col1, col2);
            }
            jpiv[i] = jpsv;
            if (abs(T16(i, i)) < smin) {
                info = 1;
                T16(i, i) = smin;
            }
            for (idx_t j = i + 1; j < 4; ++j) {
                T16(j, i) = T16(j, i) / T16(i, i);
                btmp[j] = btmp[j] - T16(j, i) * btmp[i];
                for (idx_t k = i + 1; k < 4; ++k) {
                    T16(j, k) = T16(j, k) - T16(j, i) * T16(i, k);
                }
            }
        }

        if (abs(T16(3, 3)) < smin) {
            info = 1;
            T16(3, 3) = smin;
        }
        scale = one;
        if ((eight * small_num) * abs(btmp[0]) > abs(T16(0, 0)) or
            (eight * small_num) * abs(btmp[1]) > abs(T16(1, 1)) or
            (eight * small_num) * abs(btmp[2]) > abs(T16(2, 2)) or
            (eight * small_num) * abs(btmp[3]) > abs(T16(3, 3))) {
            scale = (one / eight) / max(max(abs(btmp[0]), abs(btmp[1])),
                                        max(abs(btmp[2]), abs(btmp[3])));
            btmp[0] = btmp[0] * scale;
            btmp[1] = btmp[1] * scale;
            btmp[2] = btmp[2] * scale;
            btmp[3] = btmp[3] * scale;
        }
        for (idx_t i = 0; i < 4; ++i) {
            idx_t k = 3 - i;
            T temp = one / T16(k, k);
            tmp[k] = btmp[k] * temp;
            for (idx_t j = k + 1; j < 4; ++j) {
                tmp[k] = tmp[k] - (temp * T16(k, j)) * tmp[j];
            }
        }
        for (idx_t i = 0; i < 3; ++i) {
            if (jpiv[2 - i] != 2 - i) {
                const T temp = tmp[2 - i];
                tmp[2 - i] = tmp[jpiv[2 - i]];
                tmp[jpiv[2 - i]] = temp;
            }
        }
        X(0, 0) = tmp[0];
        X(1, 0) = tmp[1];
        X(0, 1) = tmp[2];
        X(1, 1) = tmp[3];
        xnorm = max(abs(tmp[0]) + abs(tmp[2]), abs(tmp[1]) + abs(tmp[3]));

        return info;
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LASY2_HH
