/// @file lahqr_shiftcolumn.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlaqr1.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAHQR_SHIFTCOLUMN_HH
#define TLAPACK_LAHQR_SHIFTCOLUMN_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Given a 2-by-2 or 3-by-3 matrix H, lahqr_shiftcolumn
 *  calculates a multiple of the product:
 *  (H - s1*I)*(H - s2*I)*e1
 *
 *  This is used to introduce shifts in the QR algorithm
 *
 * @return  0 if success
 *
 * @param[in] H 2x2 or 3x3 matrix.
 *      The matrix H as in the formula above.
 * @param[out] v vector of size 2 or 3
 *      On exit, a multiple of the product
 * @param[in] s1
 *      The scalar s1 as in the formula above
 * @param[in] s2
 *      The scalar s2 as in the formula above
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          enable_if_t<is_real<type_t<matrix_t>>, bool> = true>
int lahqr_shiftcolumn(const matrix_t& H,
                      vector_t& v,
                      complex_type<type_t<matrix_t>> s1,
                      complex_type<type_t<matrix_t>> s2)
{
    // Using
    using idx_t = size_type<matrix_t>;
    using TH = type_t<matrix_t>;

    // Constants
    const idx_t n = ncols(H);
    const TH zero(0);

    // Check arguments
    tlapack_check_false((n != 2 and n != 3));
    tlapack_check_false(n != nrows(H));
    tlapack_check_false((idx_t)size(v) != n);

    if (n == 2) {
        const TH s = abs(H(0, 0) - real(s2)) + abs(imag(s2)) + abs(H(1, 0));
        if (s == zero) {
            v[0] = zero;
            v[1] = zero;
        }
        else {
            const TH h10s = H(1, 0) / s;
            v[0] = h10s * H(0, 1) +
                   (H(0, 0) - real(s1)) * ((H(0, 0) - real(s2)) / s) -
                   imag(s1) * (imag(s2) / s);
            v[1] = h10s * (H(0, 0) + H(1, 1) - real(s1) - real(s2));
        }
    }
    else {
        const TH s = abs(H(0, 0) - real(s2)) + abs(imag(s2)) + abs(H(1, 0)) +
                     abs(H(2, 0));
        if (s == zero) {
            v[0] = zero;
            v[1] = zero;
            v[2] = zero;
        }
        else {
            const TH h10s = H(1, 0) / s;
            const TH h20s = H(2, 0) / s;
            v[0] = (H(0, 0) - real(s1)) * ((H(0, 0) - real(s2)) / s) -
                   imag(s1) * (imag(s2) / s) + H(0, 1) * h10s + H(0, 2) * h20s;
            v[1] = h10s * (H(0, 0) + H(1, 1) - real(s1) - real(s2)) +
                   H(1, 2) * h20s;
            v[2] = h20s * (H(0, 0) + H(2, 2) - real(s1) - real(s2)) +
                   h10s * H(2, 1);
        }
    }
    return 0;
}

/** Given a 2-by-2 or 3-by-3 matrix H, lahqr_shiftcolumn
 *  calculates a multiple of the product:
 *  (H - s1*I)*(H - s2*I)*e1
 *
 *  This is used to introduce shifts in the QR algorithm
 *
 * @return  0 if success
 *
 * @param[in] H 2x2 or 3x3 matrix.
 *      The matrix H as in the formula above.
 * @param[out] v vector of size 2 or 3
 *      On exit, a multiple of the product
 * @param[in] s1
 *      The scalar s1 as in the formula above
 * @param[in] s2
 *      The scalar s2 as in the formula above
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          enable_if_t<is_complex<type_t<matrix_t>>, bool> = true>
int lahqr_shiftcolumn(const matrix_t& H,
                      vector_t& v,
                      type_t<matrix_t> s1,
                      type_t<matrix_t> s2)
{
    // Using
    using idx_t = size_type<matrix_t>;
    using TH = type_t<matrix_t>;
    using real_t = real_type<TH>;

    // Constants
    const idx_t n = ncols(H);
    const real_t zero(0);

    // Check arguments
    tlapack_check_false((n != 2 and n != 3));
    tlapack_check_false(n != nrows(H));
    tlapack_check_false((idx_t)size(v) != n);

    if (n == 2) {
        const TH s = abs1(H(0, 0) - s2) + abs1(H(1, 0));
        if (s == zero) {
            v[0] = zero;
            v[1] = zero;
        }
        else {
            const TH h10s = H(1, 0) / s;
            v[0] = h10s * H(0, 1) + (H(0, 0) - s1) * ((H(0, 0) - s2) / s);
            v[1] = h10s * (H(0, 0) + H(1, 1) - s1 - s2);
        }
    }
    else {
        const TH s = abs1(H(0, 0) - s2) + abs1(H(1, 0)) + abs1(H(2, 0));
        if (s == zero) {
            v[0] = zero;
            v[1] = zero;
            v[2] = zero;
        }
        else {
            const TH h10s = H(1, 0) / s;
            const TH h20s = H(2, 0) / s;
            v[0] = (H(0, 0) - s1) * ((H(0, 0) - s2) / s) + H(0, 1) * h10s +
                   H(0, 2) * h20s;
            v[1] = h10s * (H(0, 0) + H(1, 1) - s1 - s2) + H(1, 2) * h20s;
            v[2] = h20s * (H(0, 0) + H(2, 2) - s1 - s2) + h10s * H(2, 1);
        }
    }
    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LAHQR_SHIFTCOLUMN_HH
