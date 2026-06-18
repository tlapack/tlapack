/// @file lahqz_schur22.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlagv2.f
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LAHQZ_SCHUR22_HH__
#define __TLAPACK_LAHQZ_SCHUR22_HH__

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"
#include "tlapack/lapack/lapy2.hpp"
#include "tlapack/lapack/laset.hpp"

namespace tlapack {

/** Computes the generalized Schur factorization of a 2x2 pencil (A,B) with B
 * upper triangular.
 *
 * (A, B) = (Q*S*Z', Q*T*Z'), where T is upper triangular, S is
 * quasi-triangular, and Q and Z are unitary.
 *
 * The Schur factorization is normalized in the following way:
 * - If the matrix is complex, or if the matrix has real eigenvalues,
 *   then S is upper triangular.
 * - The diagonal entries of T are real and non-negative.
 *
 * Note that if the matrix is real and has complex eigenvalues, then LAPACK
 * would choose to normalize the pencil by making T a diagonal matrix with
 * positive real entries. We don't do this (I just don't see why this is useful)
 *
 * @param[in,out] A 2x2 matrix
 *                On entry, the matrix A.
 *                On exit, the matrix S.
 * @param[in,out] B 2x2 upper triangular matrix
 *                On entry, the matrix B.
 *                On exit, the matrix T.
 * @param[out]    Q 2x2 unitary matrix
 *                   On exit, the matrix Q.
 * @param[out]    Z 2x2 unitary matrix
 *                   On exit, the matrix Z.
 * @param[out]    alpha1 complex number
 * @param[out]    alpha2 complex number
 * @param[out]    beta1 number
 * @param[out]    beta2 number
 *                   On exit, (alpha1, beta1), (alpha2, beta2) are the
 *                   generalized eigenvalues of the pencil (A,B)
 *
 *
 */
template <TLAPACK_MATRIX A_t,
          TLAPACK_MATRIX B_t,
          TLAPACK_MATRIX Q_t,
          TLAPACK_MATRIX Z_t>
void lahqz_schur22(A_t& A,
                   B_t& B,
                   Q_t& Q,
                   Z_t& Z,
                   complex_type<type_t<A_t>>& alpha1,
                   complex_type<type_t<A_t>>& alpha2,
                   real_type<type_t<A_t>>& beta1,
                   real_type<type_t<A_t>>& beta2)
{
    using TA = type_t<A_t>;
    using real_t = real_type<TA>;
    using complex_t = complex_type<TA>;

    real_t safmin = safe_min<real_t>();
    real_t eps = ulp<real_t>();

    //
    // Scale A
    //
    real_t anorm = max<real_t>(max<real_t>(abs1(A(0, 0)) + abs1(A(1, 0)),
                                           abs1(A(0, 1)) + abs1(A(1, 1))),
                               safmin);
    real_t ascale = real_t(1) / anorm;
    A(0, 0) *= ascale;
    A(0, 1) *= ascale;
    A(1, 0) *= ascale;
    A(1, 1) *= ascale;
    //
    // Scale B
    //
    real_t bnorm = max<real_t>(
        max<real_t>(abs1(B(0, 0)), abs1(B(0, 1)) + abs1(B(1, 1))), safmin);
    real_t bscale = real_t(1) / bnorm;
    B(0, 0) *= bscale;
    B(0, 1) *= bscale;
    B(1, 1) *= bscale;
    //
    // Check if A can be deflated
    // Note, here we deviate from LAPACK by using elementwise instead of
    // normwise.
    //
    real_t tst = abs1(A(0, 0)) + abs1(A(1, 1));
    if (abs1(A(1, 0)) <= eps * tst) {
        A(1, 0) = TA(0);
        laset(GENERAL, TA(0), TA(1), Q);
        laset(GENERAL, TA(0), TA(1), Z);
    }
    //
    // Check if B is singular
    // If it is, we can apply rotations to the right to make A(1,0) zero
    // without perturbing the upper triangular structure of B
    // Note, because of the scaling, we can just use an absolute threshold here
    //
    else if (abs1(B(0, 0)) <= eps) {
        B(0, 0) = TA(0);
        real_t c;
        TA s;
        rotg(A(0, 0), A(1, 0), c, s);
        A(1, 0) = TA(0);
        TA temp = c * A(0, 1) + s * A(1, 1);
        A(1, 1) = c * A(1, 1) - conj(s) * A(0, 1);
        A(0, 1) = temp;
        temp = c * B(0, 1) + s * B(1, 1);
        B(1, 1) = c * B(1, 1) - conj(s) * B(0, 1);
        B(0, 1) = temp;

        Q(0, 0) = c;
        Q(1, 0) = conj(s);
        Q(0, 1) = -s;
        Q(1, 1) = c;

        laset(GENERAL, TA(0), TA(1), Z);
    }
    else if (abs1(B(1, 1)) <= eps) {
        B(1, 1) = TA(0);
        real_t c;
        TA s;
        rotg(A(1, 1), A(1, 0), c, s);
        A(1, 0) = TA(0);
        TA temp = c * A(0, 1) + s * A(0, 0);
        A(0, 0) = c * A(0, 0) - conj(s) * A(0, 1);
        A(0, 1) = temp;
        temp = c * B(0, 1) + s * B(0, 0);
        B(0, 0) = c * B(0, 0) - conj(s) * B(0, 1);
        B(0, 1) = temp;

        Z(0, 0) = c;
        Z(0, 1) = s;
        Z(1, 0) = -conj(s);
        Z(1, 1) = c;

        laset(GENERAL, TA(0), TA(1), Q);
    }
    else {
        // A cannot be deflated and B is nonsingular
        // Next step is to compute the generalized eigenvalues of the pencil
        // (A,B)
        lahqz_eig22(A, B, alpha1, alpha2, beta1, beta2);
        if (is_complex<TA> or imag(alpha1) == real_t(0)) {
            // We can further reduce the pencil into 2 1x1 blocks
            // by applying rotations to the left and right

            // Compute H = beta1*A - alpha1*B
            TA h00, h01, h10, h11;
            if constexpr (is_complex<TA>) {
                h00 = beta1 * A(0, 0) - alpha1 * B(0, 0);
                h01 = beta1 * A(0, 1) - alpha1 * B(0, 1);
                h10 = beta1 * A(1, 0);
                h11 = beta1 * A(1, 1) - alpha1 * B(1, 1);
            }
            else {
                h00 = beta1 * A(0, 0) - real(alpha1) * B(0, 0);
                h01 = beta1 * A(0, 1) - real(alpha1) * B(0, 1);
                h10 = beta1 * A(1, 0);
                h11 = beta1 * A(1, 1) - real(alpha1) * B(1, 1);
            }

            real_t rr;
            if constexpr (is_complex<TA>) {
                rr = lapy2(lapy2(real(h00), imag(h00)),
                           lapy2(real(h01), imag(h01)));
            }
            else {
                rr = lapy2(h00, h01);
            }

            real_t qq;
            if constexpr (is_complex<TA>) {
                qq = lapy2(lapy2(real(h10), imag(h10)),
                           lapy2(real(h11), imag(h11)));
            }
            else {
                qq = lapy2(h10, h11);
            }

            real_t c;
            TA s;
            if (rr > qq) {
                // Find right rotation matrix to zero 0,0 element
                // of (sA - wB)
                rotg(h01, h00, c, s);
            }
            else {
                // Find right rotation matrix to zero 1,0 element
                // of (sA - wB)
                rotg(h11, h10, c, s);
            }

            // Apply the rotation to A, B, and form Z
            TA temp;
            temp = c * A(0, 1) + s * A(0, 0);
            A(0, 0) = c * A(0, 0) - conj(s) * A(0, 1);
            A(0, 1) = temp;
            temp = c * A(1, 1) + s * A(1, 0);
            A(1, 0) = c * A(1, 0) - conj(s) * A(1, 1);
            A(1, 1) = temp;
            temp = c * B(0, 1) + s * B(0, 0);
            B(0, 0) = c * B(0, 0) - conj(s) * B(0, 1);
            B(0, 1) = temp;
            B(1, 0) = -conj(s) * B(1, 1);
            B(1, 1) = c * B(1, 1);

            Z(0, 0) = c;
            Z(0, 1) = s;
            Z(1, 0) = -conj(s);
            Z(1, 1) = c;

            //
            // Now make both A and B upper triangular by applying a left
            // rotation
            //

            real_t Anrm = max<real_t>(abs1(A(0, 0)) + abs1(A(0, 1)),
                                      abs1(A(1, 0)) + abs1(A(1, 1)));
            real_t Bnrm = max<real_t>(abs1(B(0, 0)) + abs1(B(0, 1)),
                                      abs1(B(1, 0)) + abs1(B(1, 1)));
            if (Anrm >= Bnrm) {
                rotg(A(0, 0), A(1, 0), c, s);
                A(1, 0) = TA(0);
                B(0, 0) = c * B(0, 0) + s * B(1, 0);
                B(1, 0) = TA(0);
            }
            else {
                rotg(B(0, 0), B(1, 0), c, s);
                B(1, 0) = TA(0);
                A(0, 0) = c * A(0, 0) + s * A(1, 0);
                A(1, 0) = TA(0);
            }

            // Apply the rotation to A, B, and form Q
            temp = c * A(0, 1) + s * A(1, 1);
            A(1, 1) = c * A(1, 1) - conj(s) * A(0, 1);
            A(0, 1) = temp;
            temp = c * B(0, 1) + s * B(1, 1);
            B(1, 1) = c * B(1, 1) - conj(s) * B(0, 1);
            B(0, 1) = temp;

            Q(0, 0) = c;
            Q(1, 0) = conj(s);
            Q(0, 1) = -s;
            Q(1, 1) = c;
        }
        else {
            // LAPACK takes the SVD of B here, but we choose
            // to only rely on making the diagonal of B real and non-negative
            // SVD can easily be added if necessary
            laset(GENERAL, TA(0), TA(1), Q);
            laset(GENERAL, TA(0), TA(1), Z);
        }
    }

    //
    // Make sure the diagonal of B is non-negative and real
    //
    if constexpr (is_complex<TA>) {
        // Multiply A and B by [conj(B00)/|B00|, 0; 0, conj(B11)/|B11|] to
        // make B diagonal with non-negative entries
        // To avoid overflow, we use rotg to compute the angle instead
        // of the naive approach.
        real_t br = real(B(0, 0));
        real_t bi = imag(B(0, 0));
        real_t c, s;
        rotg(br, bi, c, s);
        if (br < 0) {
            br = -br;
            c = -c;
            s = -s;
        }
        TA scal0 = complex_t(c, -s);
        B(0, 0) = br;
        A(0, 0) *= scal0;
        A(1, 0) *= scal0;

        br = real(B(1, 1));
        bi = imag(B(1, 1));
        rotg(br, bi, c, s);
        if (br < 0) {
            br = -br;
            c = -c;
            s = -s;
        }
        TA scal1 = complex_t(c, -s);
        B(1, 1) = br;
        B(0, 1) *= scal1;
        A(0, 1) *= scal1;
        A(1, 1) *= scal1;

        Z(0, 0) *= scal0;
        Z(1, 0) *= scal0;
        Z(0, 1) *= scal1;
        Z(1, 1) *= scal1;
    }
    else {
        // Ensure that the diagonal of B is non-negative
        if (B(0, 0) < 0) {
            B(0, 0) = -B(0, 0);
            A(0, 0) = -A(0, 0);
            A(1, 0) = -A(1, 0);
            Z(0, 0) = -Z(0, 0);
            Z(1, 0) = -Z(1, 0);
        }
        if (B(1, 1) < 0) {
            B(0, 1) = -B(0, 1);
            B(1, 1) = -B(1, 1);
            A(0, 1) = -A(0, 1);
            A(1, 1) = -A(1, 1);
            Z(0, 1) = -Z(0, 1);
            Z(1, 1) = -Z(1, 1);
        }
    }

    // Undo scaling
    A(0, 0) *= anorm;
    A(0, 1) *= anorm;
    A(1, 0) *= anorm;
    A(1, 1) *= anorm;
    B(0, 0) *= bnorm;
    B(0, 1) *= bnorm;
    B(1, 1) *= bnorm;
    B(1, 0) *= bnorm;

    if (A(1, 0) == TA(0)) {
        alpha1 = A(0, 0);
        beta1 = real(B(0, 0));
        alpha2 = A(1, 1);
        beta2 = real(B(1, 1));
    }
    else {
        alpha1 *= anorm;
        alpha2 *= anorm;
        beta1 *= bnorm;
        beta2 *= bnorm;
    }
}

}  // namespace tlapack

#endif  // __LAHQZ_SCHUR22_HH__
