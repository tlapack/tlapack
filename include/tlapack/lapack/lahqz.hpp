/// @file lahqz.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dhgeqz.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAHQZ_HH
#define TLAPACK_LAHQZ_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"
#include "tlapack/lapack/lahqz_eig22.hpp"
#include "tlapack/lapack/lahqz_shiftcolumn.hpp"

namespace tlapack {

/** lahqz computes the eigenvalues of a matrix pair (H,T),
 *  where H is an upper Hessenberg matrix and T is upper triangular,
 *  using the single/double-shift implicit QZ algorithm.
 *
 * @return  0 if success
 * @return  i if the QR algorithm failed to compute all the eigenvalues
 *            in a total of 30 iterations per eigenvalue. elements
 *            i:ihi of w contain those eigenvalues which have been
 *            successfully computed.
 *
 * @param[in] want_s bool.
 *      If true, the full Schur form will be computed.
 * @param[in] want_q bool.
 *      If true, the Schur vectors Q will be computed.
 * @param[in] want_z bool.
 *      If true, the Schur vectors Z will be computed.
 * @param[in] ilo    integer.
 *      Either ilo=0 or A(ilo,ilo-1) = 0.
 * @param[in] ihi    integer.
 *      The matrix A is assumed to be already quasi-triangular in rows and
 *      columns ihi:n.
 * @param[in,out] A  n by n matrix.
 * @param[out] alpha  size n vector.
 * @param[out] beta  size n vector.
 * @param[in,out] Q  n by n matrix.
 * @param[in,out] Z  n by n matrix.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_CSMATRIX matrix_t,
          TLAPACK_VECTOR alpha_t,
          TLAPACK_VECTOR beta_t>
int lahqz(bool want_s,
          bool want_q,
          bool want_z,
          size_type<matrix_t> ilo,
          size_type<matrix_t> ihi,
          matrix_t& A,
          matrix_t& B,
          alpha_t& alpha,
          beta_t& beta,
          matrix_t& Q,
          matrix_t& Z)
{
    using TA = type_t<matrix_t>;
    using real_t = real_type<TA>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Functor
    Create<vector_type<matrix_t>> new_vector;

    // constants
    const real_t zero(0);
    const real_t one(1);
    const real_t eps = ulp<real_t>();
    const real_t small_num = safe_min<real_t>() / ulp<real_t>();
    const idx_t non_convergence_limit = 10;
    const real_t dat1(0.75);
    const real_t dat2(-0.4375);

    const idx_t n = ncols(A);
    const idx_t nh = ihi - ilo;

    // check arguments
    tlapack_check_false(n != nrows(A));
    tlapack_check_false(n != ncols(B));
    tlapack_check_false(n != nrows(B));
    tlapack_check_false((idx_t)size(alpha) != n);
    tlapack_check_false((idx_t)size(beta) != n);
    if (want_q) {
        tlapack_check_false((n != ncols(Q)) or (n != nrows(Q)));
    }
    if (want_z) {
        tlapack_check_false((n != ncols(Z)) or (n != nrows(Z)));
    }

    // quick return
    if (nh <= 0) return 0;
    if (nh == 1) {
        alpha[ilo] = A(ilo, ilo);
        beta[ilo] = B(ilo, ilo);
    }

    // itmax is the total number of QR iterations allowed.
    // For most matrices, 3 shifts per eigenvalue is enough, so
    // we set itmax to 30 times nh as a safe limit.
    const idx_t itmax = 30 * std::max<idx_t>(10, nh);

    // k_defl counts the number of iterations since a deflation
    idx_t k_defl = 0;

    // istop is the end of the active subblock.
    // As more and more eigenvalues converge, it eventually
    // becomes ilo+1 and the loop ends.
    idx_t istop = ihi;
    // istart is the start of the active subblock. Either
    // istart = ilo, or H(istart, istart-1) = 0. This means
    // that we can treat this subblock separately.
    idx_t istart = ilo;

    for (idx_t iter = 0; iter <= itmax; ++iter) {
        if (iter == itmax) {
            // The QZ algorithm failed to converge, return with error.
            tlapack_error(
                istop,
                "The QR algorithm failed to compute all the eigenvalues"
                " in a total of 30 iterations per eigenvalue. Elements"
                " i:ihi of w contain those eigenvalues which have been"
                " successfully computed.");
            return istop;
        }

        if (istart + 1 >= istop) {
            if (istart + 1 == istop) {
                alpha[istart] = A(istart, istart);
                beta[istart] = B(istart, istart);
            }
            // All eigenvalues have been found, exit and return 0.
            break;
        }

        // Determine range to apply rotations
        idx_t istart_m;
        idx_t istop_m;
        if (!want_s) {
            istart_m = istart;
            istop_m = istop;
        }
        else {
            istart_m = 0;
            istop_m = n;
        }

        // Check if active subblock has split
        for (idx_t i = istop - 1; i > istart; --i) {
            if (abs1(A(i, i - 1)) <= small_num) {
                // A(i,i-1) is negligible, take i as new istart.
                A(i, i - 1) = zero;
                istart = i;
                break;
            }

            real_t tst = abs1(A(i - 1, i - 1)) + abs1(A(i, i));
            if (tst == zero) {
                if (i >= ilo + 2) {
                    tst = tst + abs(A(i - 1, i - 2));
                }
                if (i < ihi) {
                    tst = tst + abs(A(i + 1, i));
                }
            }
            if (abs1(A(i, i - 1)) <= eps * tst) {
                //
                // The elementwise deflation test has passed
                // The following performs second deflation test due
                // to Steel, Vandebril and Langou (2023). It has better
                // mathematical foundation and improves accuracy in some
                // examples.
                //
                // The test is |A(i,i-1)|*|A(i-1,i)*B(i,i) - A(i,i)*B(i-1,i)| <=
                // eps*|A(i,i)|*|A(i-1,i-1)*B(i,i)-A(i,i)*B(i-1,i-1)|.
                // The multiplications might overflow so we do some scaling
                // first.
                //
                const real_t tst1 =
                    abs1(B(i, i) * A(i - 1, i) - A(i, i) * B(i - 1, i));
                const real_t tst2 =
                    abs1(B(i, i) * A(i - 1, i - 1) - A(i, i) * B(i - 1, i - 1));
                const real_t aij = abs1(A(i, i - 1));
                real_t ab = max(abs1(A(i, i - 1)), tst1);
                real_t ba = min(abs1(A(i, i - 1)), tst1);
                real_t aa = max(abs1(A(i, i)), tst2);
                real_t bb = min(abs1(A(i, i)), tst2);
                real_t s = aa + ab;
                if (ba * (ab / s) <= max(small_num, eps * (bb * (aa / s)))) {
                    // A(i,i-1) is negligible, take i as new istart.
                    A(i, i - 1) = zero;
                    istart = i;
                    break;
                }
            }
        }

        // TODO: deflate infinite eigenvalues

        if (istart + 2 >= istop) {
            if (istart + 1 == istop) {
                // 1x1 block
                k_defl = 0;
                alpha[istart] = A(istart, istart);
                beta[istart] = B(istart, istart);
                istop = istart;
                istart = ilo;
                continue;
            }
            if constexpr (is_real<TA>) {
                if (istart + 2 == istop) {
                    // 2x2 block, normalize the block
                    // TODO!
                    k_defl = 0;
                    istop = istart;
                    istart = ilo;
                    continue;
                }
            }
        }

        // Determine shift
        k_defl = k_defl + 1;

        complex_type<real_t> shift1;
        complex_type<real_t> shift2;
        real_t beta1;
        real_t beta2;
        if (k_defl % non_convergence_limit == 0) {
            // Exceptional shift
            // TODO!
        }
        else {
            // Wilkinson shift
            auto A22 =
                slice(A, range(istop - 2, istop), range(istop - 2, istop));
            auto B22 =
                slice(B, range(istop - 2, istop), range(istop - 2, istop));
            lahqz_eig22(A22, B22, shift1, shift2, beta1, beta2);
            if ((imag(shift1) == zero and imag(shift2) == zero) or
                is_complex<TA>) {
                // The eigenvalues are not complex conjugate, keep only the one
                // closest to A(istop-1, istop-1)
                if (abs1(B(istop - 1, istop - 1) * shift1 -
                         beta1 * A(istop - 1, istop - 1)) <=
                    abs1(B(istop - 1, istop - 1) * shift2 -
                         beta2 * A(istop - 1, istop - 1))) {
                    shift2 = shift1;
                    beta2 = beta1;
                }
                else {
                    shift1 = shift2;
                    beta1 = beta2;
                }
            }
        }

        // We have already checked whether the subblock has split.
        // If it has split, we can introduce any shift at the top of the new
        // subblock. Now that we know the specific shift, we can also check
        // whether we can introduce that shift somewhere else in the subblock.
        TA t1;
        idx_t istart2 = istart;
        std::vector<TA> v_;
        auto v = new_vector(v_, 3);
        // TODO

        // All the preparations are done, we can apply an implicit QZ iteration
        for (idx_t i = istart2; i < istop - 1; ++i) {
            const idx_t nr = std::min<idx_t>(3, istop - i);
            real_t c1, c2;
            TA s1, s2;

            // Calculate rotations from the left
            if (i == istart2) {
                auto H = slice(A, range{i, i + nr}, range{i, i + nr});
                auto T = slice(B, range{i, i + nr}, range{i, i + nr});
                auto x = slice(v, range{0, nr});
                lahqz_shiftcolumn(H, T, x, shift1, shift2, beta1, beta2);

                if (nr == 3) rotg(x[1], x[2], c1, s1);
                rotg(x[0], x[1], c2, s2);

                if (i > istart) {
                    // TODO
                    // A(i, i - 1) = A(i, i - 1) * (one - conj(t1));
                }
            }
            else {
                if (nr == 3) {
                    rotg(A(i + 1, i - 1), A(i + 2, i - 1), c1, s1);
                    A(i + 2, i - 1) = zero;
                }
                rotg(A(i, i - 1), A(i + 1, i - 1), c2, s2);
                A(i + 1, i - 1) = zero;
            }
            // Apply rotations from the left
            if (nr == 3) {
                auto a1 = slice(A, i, range{i, istop_m});
                auto a2 = slice(A, i + 1, range{i, istop_m});
                auto a3 = slice(A, i + 2, range{i, istop_m});
                rot(a2, a3, c1, s1);
                rot(a1, a2, c2, s2);
                auto b1 = slice(B, i, range{i, istop_m});
                auto b2 = slice(B, i + 1, range{i, istop_m});
                auto b3 = slice(B, i + 2, range{i, istop_m});
                rot(b2, b3, c1, s1);
                rot(b1, b2, c2, s2);
                if (want_q) {
                    auto q1 = col(Q, i);
                    auto q2 = col(Q, i + 1);
                    auto q3 = col(Q, i + 2);
                    rot(q2, q3, c1, conj(s1));
                    rot(q1, q2, c2, conj(s2));
                }
            }
            else {
                auto a1 = slice(A, i, range{i, istop_m});
                auto a2 = slice(A, i + 1, range{i, istop_m});
                rot(a1, a2, c2, s2);
                auto b1 = slice(B, i, range{i, istop_m});
                auto b2 = slice(B, i + 1, range{i, istop_m});
                rot(b1, b2, c2, s2);
                if (want_q) {
                    auto q1 = col(Q, i);
                    auto q2 = col(Q, i + 1);
                    rot(q1, q2, c2, conj(s2));
                }
            }

            // Remove fill-in from B
            if (nr == 3) {
                rotg(B(i + 2, i + 2), B(i + 2, i + 1), c1, s1);
                s1 = -s1;
                B(i + 2, i + 1) = 0.;
                // Also apply the rotation to the 2 elements above so that
                // we can calculate the next rotation
                auto temp = c1 * B(i + 1, i + 1) + conj(s1) * B(i + 1, i + 2);
                B(i + 1, i + 2) = -s1 * B(i + 1, i + 1) + c1 * B(i + 1, i + 2);
                B(i + 1, i + 1) = temp;
            }
            rotg(B(i + 1, i + 1), B(i + 1, i), c2, s2);
            s2 = -s2;
            B(i + 1, i) = 0.;

            // Apply rotation from the right
            if (nr == 3) {
                auto b1 = slice(B, range{istart_m, i + 1}, i);
                auto b2 = slice(B, range{istart_m, i + 1}, i + 1);
                auto b3 = slice(B, range{istart_m, i + 1}, i + 2);
                rot(b2, b3, c1, conj(s1));
                rot(b1, b2, c2, conj(s2));
                auto a1 = slice(A, range{istart_m, min(i + 4, ihi)}, i);
                auto a2 = slice(A, range{istart_m, min(i + 4, ihi)}, i + 1);
                auto a3 = slice(A, range{istart_m, min(i + 4, ihi)}, i + 2);
                rot(a2, a3, c1, conj(s1));
                rot(a1, a2, c2, conj(s2));
                if (want_z) {
                    auto z1 = col(Z, i);
                    auto z2 = col(Z, i + 1);
                    auto z3 = col(Z, i + 2);
                    rot(z2, z3, c1, conj(s1));
                    rot(z1, z2, c2, conj(s2));
                }
            }
            else {
                auto b1 = slice(B, range{istart_m, i + 1}, i);
                auto b2 = slice(B, range{istart_m, i + 1}, i + 1);
                rot(b1, b2, c2, conj(s2));
                auto a1 = slice(A, range{istart_m, min(i + 4, ihi)}, i);
                auto a2 = slice(A, range{istart_m, min(i + 4, ihi)}, i + 1);
                rot(a1, a2, c2, conj(s2));
                if (want_z) {
                    auto z1 = col(Z, i);
                    auto z2 = col(Z, i + 1);
                    rot(z1, z2, c2, conj(s2));
                }
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LAHQZ_HH
