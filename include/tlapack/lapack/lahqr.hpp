/// @file lahqr.hpp
/// @author Thijs, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlahqr.f
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAHQR_HH
#define TLAPACK_LAHQR_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/lahqr_eig22.hpp"
#include "tlapack/lapack/lahqr_shiftcolumn.hpp"
#include "tlapack/lapack/lahqr_schur22.hpp"

namespace tlapack
{

    /** lahqr computes the eigenvalues and optionally the Schur
     *  factorization of an upper Hessenberg matrix, using the double-shift
     *  implicit QR algorithm.
     *
     *  The Schur factorization is returned in standard form. For complex matrices
     *  this means that the matrix T is upper-triangular. The diagonal entries
     *  of T are also its eigenvalues. For real matrices, this means that the
     *  matrix T is block-triangular, with real eigenvalues appearing as 1x1 blocks
     *  on the diagonal and imaginary eigenvalues appearing as 2x2 blocks on the diagonal.
     *  All 2x2 blocks are normalized so that the diagonal entries are equal to the real part
     *  of the eigenvalue.
     *
     *
     * @return  0 if success
     * @return  i if the QR algorithm failed to compute all the eigenvalues
     *            in a total of 30 iterations per eigenvalue. elements
     *            i:ihi of w contain those eigenvalues which have been
     *            successfully computed.
     *
     * @param[in] want_t bool.
     *      If true, the full Schur factor T will be computed.
     * @param[in] want_z bool.
     *      If true, the Schur vectors Z will be computed.
     * @param[in] ilo    integer.
     *      Either ilo=0 or A(ilo,ilo-1) = 0.
     * @param[in] ihi    integer.
     *      The matrix A is assumed to be already quasi-triangular in rows and
     *      columns ihi:n.
     * @param[in,out] A  n by n matrix.
     *      On entry, the matrix A.
     *      On exit, if info=0 and want_t=true, the Schur factor T.
     *      T is quasi-triangular in rows and columns ilo:ihi, with
     *      the diagonal (block) entries in standard form (see above).
     * @param[out] w  size n vector.
     *      On exit, if info=0, w(ilo:ihi) contains the eigenvalues
     *      of A(ilo:ihi,ilo:ihi). The eigenvalues appear in the same
     *      order as the diagonal (block) entries of T.
     * @param[in,out] Z  n by n matrix.
     *      On entry, the previously calculated Schur factors
     *      On exit, the orthogonal updates applied to A are accumulated
     *      into Z.
     *
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true,
        enable_if_t<!is_complex<type_t<matrix_t>>::value, bool> = true>
    int lahqr(bool want_t, bool want_z, size_type<matrix_t> ilo, size_type<matrix_t> ihi, matrix_t &A, vector_t &w, matrix_t &Z)
    {
        using TA = type_t<matrix_t>;
        using real_t = real_type<TA>;
        using idx_t = size_type<matrix_t>;
        using pair = pair<idx_t, idx_t>;

        // constants
        const real_t rzero(0);
        const TA one(1);
        const TA zero(0);
        const real_t eps = ulp<real_t>();
        const real_t small_num = safe_min<real_t>() / ulp<real_t>();
        const idx_t non_convergence_limit = 10;
        const real_t dat1 = 3.0 / 4.0;
        const real_t dat2 = -0.4375;

        const idx_t n = ncols(A);
        const idx_t nh = ihi - ilo;

        // check arguments
        tlapack_check_false(n != nrows(A) );
        tlapack_check_false((idx_t)size(w) != n );
        if (want_z)
        {
            tlapack_check_false((n != ncols(Z)) or (n != nrows(Z)) );
        }

        // quick return
        if (nh <= 0)
            return 0;
        if (nh == 1)
            w[ilo] = A(ilo, ilo);

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

        for (idx_t iter = 0; iter <= itmax; ++iter)
        {
            if (iter == itmax)
            {
                // The QR algorithm failed to converge, return with error.
                tlapack_error( istop,
                    "The QR algorithm failed to compute all the eigenvalues"
                    " in a total of 30 iterations per eigenvalue. Elements"
                    " i:ihi of w contain those eigenvalues which have been"
                    " successfully computed." );
                return istop;
            }

            if (istart + 1 >= istop)
            {
                if (istart + 1 == istop)
                    w[istart] = A(istart, istart);
                // All eigenvalues have been found, exit and return 0.
                break;
            }

            // Determine range to apply rotations
            idx_t istart_m;
            idx_t istop_m;
            if (!want_t)
            {
                istart_m = istart;
                istop_m = istop;
            }
            else
            {
                istart_m = 0;
                istop_m = n;
            }

            // Check if active subblock has split
            for (idx_t i = istop - 1; i > istart; --i)
            {

                if (abs1(A(i, i - 1)) <= small_num)
                {
                    // A(i,i-1) is negligible, take i as new istart.
                    A(i, i - 1) = zero;
                    istart = i;
                    break;
                }

                real_t tst = abs1(A(i - 1, i - 1)) + abs1(A(i, i));
                if (tst == rzero)
                {
                    if (i >= ilo + 2)
                    {
                        tst = tst + abs(A(i - 1, i - 2));
                    }
                    if (i < ihi)
                    {
                        tst = tst + abs(A(i + 1, i));
                    }
                }
                if (abs1(A(i, i - 1)) <= eps * tst)
                {
                    //
                    // The elementwise deflation test has passed
                    // The following performs second deflation test due
                    // to Ahues & Tisseur (LAWN 122, 1997). It has better
                    // mathematical foundation and improves accuracy in some
                    // examples.
                    //
                    // The test is |A(i,i-1)|*|A(i-1,i)| <= eps*|A(i,i)|*|A(i-1,i-1)|
                    // The multiplications might overflow so we do some scaling first.
                    //
                    real_t ab = std::max(abs1(A(i, i - 1)), abs1(A(i - 1, i)));
                    real_t ba = std::min(abs1(A(i, i - 1)), abs1(A(i - 1, i)));
                    real_t aa = std::max(abs1(A(i, i)), abs1(A(i, i) - A(i - 1, i - 1)));
                    real_t bb = std::min(abs1(A(i, i)), abs1(A(i, i) - A(i - 1, i - 1)));
                    real_t s = aa + ab;
                    if (ba * (ab / s) <= std::max(small_num, eps * (bb * (aa / s))))
                    {
                        // A(i,i-1) is negligible, take i as new istart.
                        A(i, i - 1) = zero;
                        istart = i;
                        break;
                    }
                }
            }

            if (istart + 2 >= istop)
            {
                if (istart + 1 == istop)
                {
                    // 1x1 block
                    k_defl = 0;
                    w[istart] = A(istart, istart);
                    istop = istart;
                    istart = ilo;
                    continue;
                }
                if (!is_complex<TA>::value && istart + 2 == istop)
                {
                    // 2x2 block, normalize the block
                    real_t cs;
                    TA sn;
                    // We don't check the error flag here because it should never fail for real values.
                    lahqr_schur22(A(istart, istart), A(istart, istart + 1),
                                  A(istart + 1, istart), A(istart + 1, istart + 1),
                                  w[istart], w[istart + 1], cs, sn);
                    // Apply the rotations from the normalization to the rest of the matrix.
                    if (want_t)
                    {
                        if (istart + 2 < istop_m)
                        {
                            auto x = slice(A, istart, pair{istart + 2, istop_m});
                            auto y = slice(A, istart + 1, pair{istart + 2, istop_m});
                            rot(x, y, cs, sn);
                        }
                        auto x2 = slice(A, pair{istart_m, istart}, istart);
                        auto y2 = slice(A, pair{istart_m, istart}, istart + 1);
                        rot(x2, y2, cs, sn);
                    }
                    if (want_z)
                    {
                        auto x = col(Z, istart);
                        auto y = col(Z, istart + 1);
                        rot(x, y, cs, sn);
                    }
                    k_defl = 0;
                    istop = istart;
                    istart = ilo;
                    continue;
                }
            }

            // Determine shift
            TA a00, a01, a10, a11;
            k_defl = k_defl + 1;
            if (k_defl % non_convergence_limit == 0)
            {
                // Exceptional shift
                auto s = abs(A(istop - 1, istop - 2));
                if (istop > ilo + 2)
                    s = s + abs(A(istop - 2, istop - 3));
                a00 = dat1 * s + A(istop - 1, istop - 1);
                a01 = dat2 * s;
                a10 = s;
                a11 = a00;
            }
            else
            {
                // Wilkinson shift
                a00 = A(istop - 2, istop - 2);
                a10 = A(istop - 1, istop - 2);
                a01 = A(istop - 2, istop - 1);
                a11 = A(istop - 1, istop - 1);
            }
            std::complex<real_t> s1;
            std::complex<real_t> s2;
            lahqr_eig22(a00, a01, a10, a11, s1, s2);
            if ((imag(s1) == rzero and imag(s2) == rzero) or is_complex<TA>::value)
            {
                // The eigenvalues are not complex conjugate, keep only the one closest to A(istop-1, istop-1)
                if (abs1(s1 - A(istop - 1, istop - 1)) <= abs1(s2 - A(istop - 1, istop - 1)))
                    s2 = s1;
                else
                    s1 = s2;
            }

            // We have already checked whether the subblock has split.
            // If it has split, we can introduce any shift at the top of the new subblock.
            // Now that we know the specific shift, we can also check whether we can introduce that shift
            // somewhere else in the subblock.
            std::vector<TA> v(3);
            TA t1;
            auto istart2 = istart;
            if (istart + 3 < istop)
            {
                for (idx_t i = istop - 3; i > istart; --i)
                {
                    auto H = slice(A, pair{i, i + 3}, pair{i, i + 3});
                    lahqr_shiftcolumn(H, v, s1, s2);
                    larfg(v, t1);
                    v[0] = t1;
                    auto refsum = conj(v[0]) * A(i, i - 1) + conj(v[1]) * A(i + 1, i - 1);
                    if (abs1(A(i + 1, i - 1) - refsum * v[1]) + abs1(refsum * v[2]) <=
                        eps * (abs1(A(i, i - 1)) + abs1(A(i, i + 1)) + abs1(A(i + 1, i + 2))))
                    {
                        istart2 = i;
                        break;
                    }
                }
            }

            for (idx_t i = istart2; i < istop - 1; ++i)
            {
                auto nr = std::min<idx_t>(3, istop - i);
                if (i == istart2)
                {
                    auto H = slice(A, pair{i, i + nr}, pair{i, i + nr});
                    auto x = slice(v, pair{0, nr});
                    lahqr_shiftcolumn(H, x, s1, s2);
                    x = slice(v, pair{1, nr});
                    larfg(v[0], x, t1);
                    if (i > istart)
                    {
                        A(i, i - 1) = A(i, i - 1) * (one - conj(t1));
                    }
                }
                else
                {
                    v[0] = A(i, i - 1);
                    v[1] = A(i + 1, i - 1);
                    if (nr == 3)
                        v[2] = A(i + 2, i - 1);
                    auto x = slice(v, pair{1, nr});
                    larfg(v[0], x, t1);
                    A(i, i - 1) = v[0];
                    A(i + 1, i - 1) = zero;
                    if (nr == 3)
                        A(i + 2, i - 1) = zero;
                }

                // The following code applies the reflector we have just calculated.
                // We write this out instead of using larf because a direct loop is more
                // efficient for small reflectors.

                t1 = conj(t1);
                auto v2 = v[1];
                auto t2 = t1 * v2;
                TA sum;
                if (nr == 3)
                {
                    auto v3 = v[2];
                    auto t3 = t1 * v[2];

                    // Apply G from the left to A
                    for (idx_t j = i; j < istop_m; ++j)
                    {
                        sum = A(i, j) + conj(v2) * A(i + 1, j) + conj(v3) * A(i + 2, j);
                        A(i, j) = A(i, j) - sum * t1;
                        A(i + 1, j) = A(i + 1, j) - sum * t2;
                        A(i + 2, j) = A(i + 2, j) - sum * t3;
                    }
                    // Apply G from the right to A
                    for (idx_t j = istart_m; j < std::min(i + 4, istop); ++j)
                    {
                        sum = A(j, i) + v2 * A(j, i + 1) + v3 * A(j, i + 2);
                        A(j, i) = A(j, i) - sum * conj(t1);
                        A(j, i + 1) = A(j, i + 1) - sum * conj(t2);
                        A(j, i + 2) = A(j, i + 2) - sum * conj(t3);
                    }
                    if (want_z)
                    {
                        // Apply G to Z from the right
                        for (idx_t j = 0; j < n; ++j)
                        {
                            sum = Z(j, i) + v2 * Z(j, i + 1) + v3 * Z(j, i + 2);
                            Z(j, i) = Z(j, i) - sum * conj(t1);
                            Z(j, i + 1) = Z(j, i + 1) - sum * conj(t2);
                            Z(j, i + 2) = Z(j, i + 2) - sum * conj(t3);
                        }
                    }
                }
                else
                {
                    // Apply G from the left to A
                    for (idx_t j = i; j < istop_m; ++j)
                    {
                        sum = A(i, j) + conj(v2) * A(i + 1, j);
                        A(i, j) = A(i, j) - sum * t1;
                        A(i + 1, j) = A(i + 1, j) - sum * t2;
                    }
                    // Apply G from the right to A
                    for (idx_t j = istart_m; j < std::min(i + 3, istop); ++j)
                    {
                        sum = A(j, i) + v2 * A(j, i + 1);
                        A(j, i) = A(j, i) - sum * conj(t1);
                        A(j, i + 1) = A(j, i + 1) - sum * conj(t2);
                    }
                    if (want_z)
                    {
                        // Apply G to Z from the right
                        for (idx_t j = 0; j < n; ++j)
                        {
                            sum = Z(j, i) + v2 * Z(j, i + 1);
                            Z(j, i) = Z(j, i) - sum * conj(t1);
                            Z(j, i + 1) = Z(j, i + 1) - sum * conj(t2);
                        }
                    }
                }
            }
        }

        return 0;
    }

    /** lahqr computes the eigenvalues and optionally the Schur
     *  factorization of an upper Hessenberg matrix, using the double-shift
     *  implicit QR algorithm.
     *
     *  The Schur factorization is returned in standard form. For complex matrices
     *  this means that the matrix T is upper-triangular. The diagonal entries
     *  of T are also its eigenvalues. For real matrices, this means that the
     *  matrix T is block-triangular, with real eigenvalues appearing as 1x1 blocks
     *  on the diagonal and imaginary eigenvalues appearing as 2x2 blocks on the diagonal.
     *  All 2x2 blocks are normalized so that the diagonal entries are equal to the real part
     *  of the eigenvalue.
     *
     *
     * @return  0 if success
     * @return -i if the ith argument is invalid
     * @return  i if the QR algorithm failed to compute all the eigenvalues
     *            in a total of 30 iterations per eigenvalue. elements
     *            i:ihi of w contain those eigenvalues which have been
     *            successfully computed.
     *
     * @param[in] want_t bool.
     *      If true, the full Schur factor T will be computed.
     * @param[in] want_z bool.
     *      If true, the Schur vectors Z will be computed.
     * @param[in] ilo    integer.
     *      Either ilo=0 or A(ilo,ilo-1) = 0.
     * @param[in] ihi    integer.
     *      The matrix A is assumed to be already quasi-triangular in rows and
     *      columns ihi:n.
     * @param[in,out] A  n by n matrix.
     *      On entry, the matrix A.
     *      On exit, if info=0 and want_t=true, the Schur factor T.
     *      T is quasi-triangular in rows and columns ilo:ihi, with
     *      the diagonal (block) entries in standard form (see above).
     * @param[out] w  size n vector.
     *      On exit, if info=0, w(ilo:ihi) contains the eigenvalues
     *      of A(ilo:ihi,ilo:ihi). The eigenvalues appear in the same
     *      order as the diagonal (block) entries of T.
     * @param[in,out] Z  n by n matrix.
     *      On entry, the previously calculated Schur factors
     *      On exit, the orthogonal updates applied to A are accumulated
     *      into Z.
     *
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true,
        enable_if_t<is_complex<type_t<matrix_t>>::value, bool> = true>
    int lahqr(bool want_t, bool want_z, size_type<matrix_t> ilo, size_type<matrix_t> ihi, matrix_t &A, vector_t &w, matrix_t &Z)
    {
        using TA = type_t<matrix_t>;
        using real_t = real_type<TA>;
        using idx_t = size_type<matrix_t>;

        // constants
        const real_t rzero(0);
        const TA zero(0);
        const real_t eps = ulp<real_t>();
        const real_t small_num = safe_min<real_t>() / ulp<real_t>();
        const idx_t non_convergence_limit = 10;
        const real_t dat1 = 3.0 / 4.0;
        const real_t dat2 = -0.4375;

        const idx_t n = ncols(A);
        const idx_t nh = ihi - ilo;

        // check arguments
        tlapack_check_false(n != nrows(A) );
        tlapack_check_false((idx_t)size(w) != n );
        if (want_z)
        {
            tlapack_check_false((n != ncols(Z)) or (n != nrows(Z)) );
        }

        // quick return
        if (nh <= 0)
            return 0;
        if (nh == 1)
            w[ilo] = A(ilo, ilo);

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

        for (idx_t iter = 0; iter <= itmax; ++iter)
        {
            if (iter == itmax)
            {
                // The QR algorithm failed to converge, return with error.
                return istop;
            }

            if (istart + 1 >= istop)
            {
                if (istart + 1 == istop)
                    w[istart] = A(istart, istart);
                // All eigenvalues have been found, exit and return 0.
                break;
            }

            // Determine range to apply rotations
            idx_t istart_m;
            idx_t istop_m;
            if (!want_t)
            {
                istart_m = istart;
                istop_m = istop;
            }
            else
            {
                istart_m = 0;
                istop_m = n;
            }

            // Check if active subblock has split
            for (idx_t i = istop - 1; i > istart; --i)
            {

                if (abs1(A(i, i - 1)) <= small_num)
                {
                    // A(i,i-1) is negligible, take i as new istart.
                    A(i, i - 1) = zero;
                    istart = i;
                    break;
                }

                real_t tst = abs1(A(i - 1, i - 1)) + abs1(A(i, i));
                if (tst == rzero)
                {
                    if (i >= ilo + 2)
                    {
                        tst = tst + abs(A(i - 1, i - 2));
                    }
                    if (i < ihi)
                    {
                        tst = tst + abs(A(i + 1, i));
                    }
                }
                if (abs1(A(i, i - 1)) <= eps * tst)
                {
                    //
                    // The elementwise deflation test has passed
                    // The following performs second deflation test due
                    // to Ahues & Tisseur (LAWN 122, 1997). It has better
                    // mathematical foundation and improves accuracy in some
                    // examples.
                    //
                    // The test is |A(i,i-1)|*|A(i-1,i)| <= eps*|A(i,i)|*|A(i-1,i-1)|
                    // The multiplications might overflow so we do some scaling first.
                    //
                    real_t ab = std::max(abs1(A(i, i - 1)), abs1(A(i - 1, i)));
                    real_t ba = std::min(abs1(A(i, i - 1)), abs1(A(i - 1, i)));
                    real_t aa = std::max(abs1(A(i, i)), abs1(A(i, i) - A(i - 1, i - 1)));
                    real_t bb = std::min(abs1(A(i, i)), abs1(A(i, i) - A(i - 1, i - 1)));
                    real_t s = aa + ab;
                    if (ba * (ab / s) <= std::max(small_num, eps * (bb * (aa / s))))
                    {
                        // A(i,i-1) is negligible, take i as new istart.
                        A(i, i - 1) = zero;
                        istart = i;
                        break;
                    }
                }
            }

            if (istart + 1 >= istop)
            {
                k_defl = 0;
                w[istart] = A(istart, istart);
                istop = istart;
                istart = ilo;
                continue;
            }

            // Determine shift
            TA a00, a01, a10, a11;
            k_defl = k_defl + 1;
            if (k_defl % non_convergence_limit == 0)
            {
                // Exceptional shift
                auto s = abs(A(istop - 1, istop - 2));
                if (istop > ilo + 2)
                    s = s + abs(A(istop - 2, istop - 3));
                a00 = dat1 * s + A(istop - 1, istop - 1);
                a01 = dat2 * s;
                a10 = s;
                a11 = a00;
            }
            else
            {
                // Wilkinson shift
                a00 = A(istop - 2, istop - 2);
                a10 = A(istop - 1, istop - 2);
                a01 = A(istop - 2, istop - 1);
                a11 = A(istop - 1, istop - 1);
            }
            std::complex<real_t> s1;
            std::complex<real_t> s2;
            lahqr_eig22(a00, a01, a10, a11, s1, s2);
            if (abs1(s1 - A(istop - 1, istop - 1)) > abs1(s2 - A(istop - 1, istop - 1)))
                s1 = s2;

            // We have already checked whether the subblock has split.
            // If it has split, we can introduce any shift at the top of the new subblock.
            // Now that we know the specific shift, we can also check whether we can introduce that shift
            // somewhere else in the subblock.
            TA sn;
            real_t cs;
            auto istart2 = istart;
            if (istart + 2 < istop)
            {
                for (idx_t i = istop - 2; i > istart; --i)
                {
                    auto h00 = A(i, i) - s1;
                    auto h10 = A(i + 1, i);
                    rotg(h00, h10, cs, sn);

                    if (abs1(conj(sn) * A(i, i - 1)) <= eps * (abs1(A(i, i - 1)) + abs1(A(i, i + 1))))
                    {
                        istart2 = i;
                        break;
                    }
                }
            }

            for (idx_t i = istart2; i < istop - 1; ++i)
            {
                if (i == istart2)
                {
                    auto h00 = A(i, i) - s1;
                    auto h10 = A(i + 1, i);
                    rotg(h00, h10, cs, sn);
                    if (i > istart)
                        A(i, i - 1) = A(i, i - 1) * cs;
                }
                else
                {
                    rotg(A(i, i - 1), A(i + 1, i - 1), cs, sn);
                    A(i + 1, i - 1) = zero;
                }

                // Apply G from the left to A
                for (idx_t j = i; j < istop_m; ++j)
                {
                    TA tmp = cs * A(i, j) + sn * A(i + 1, j);
                    A(i + 1, j) = -conj(sn) * A(i, j) + cs * A(i + 1, j);
                    A(i, j) = tmp;
                }
                // Apply G**H from the right to A
                for (idx_t j = istart_m; j < std::min(i + 3, istop); ++j)
                {
                    TA tmp = cs * A(j, i) + conj(sn) * A(j, i + 1);
                    A(j, i + 1) = -sn * A(j, i) + cs * A(j, i + 1);
                    A(j, i) = tmp;
                }
                if (want_z)
                {
                    // Apply G**H to Z from the right
                    for (idx_t j = 0; j < n; ++j)
                    {
                        TA tmp = cs * Z(j, i) + conj(sn) * Z(j, i + 1);
                        Z(j, i + 1) = -sn * Z(j, i) + cs * Z(j, i + 1);
                        Z(j, i) = tmp;
                    }
                }
            }
        }

        return 0;
    }

} // lapack

#endif // TLAPACK_LAHQR_HH
