/// @file lahqz_shiftcolumn.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlaqz1.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAHQZ_SHIFTCOLUMN_HH
#define TLAPACK_LAHQZ_SHIFTCOLUMN_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Given a 2-by-2 or 3-by-3 matrix pencil (A,B), lahqz_shiftcolumn
 *  calculates a multiple of the product:
 *  (beta2*A - s2*B)*B^(-1)*(beta1*A - s1*B)*B^(-1)*e1
 *
 *  This is used to introduce shifts in the QZ algorithm
 *
 * @return  0 if success
 *
 * @param[in] A 2x2 or 3x3 matrix.
 *      The matrix A as in the formula above.
 * @param[in] B 2x2 or 3x3 matrix.
 *      The matrix B as in the formula above.
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
int lahqz_shiftcolumn(const matrix_t& A,
                      const matrix_t& B,
                      vector_t& v,
                      complex_type<type_t<matrix_t>> s1,
                      complex_type<type_t<matrix_t>> s2,
                      real_type<type_t<matrix_t>> beta1,
                      real_type<type_t<matrix_t>> beta2)
{
    // Using
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // Constants
    const idx_t n = ncols(A);
    const T zero(0);
    const real_t safmin = safe_min<real_t>();
    const real_t safmax = safe_max<real_t>();

    // Check arguments
    tlapack_check_false((n != 2 and n != 3));
    tlapack_check_false(n != nrows(A));
    tlapack_check_false((idx_t)size(v) != n);

    if (n == 2) {
        T w1 = beta1 * A(0, 0) - real(s1) * B(0, 0);
        T w2 = beta1 * A(1, 0) - real(s1) * B(1, 0);
        v[0] = w1;
        v[1] = w2;
    }
    else {
        // Calculate first shifted vector
        T w1 = beta1 * A(0, 0) - real(s1) * B(0, 0);
        T w2 = beta1 * A(1, 0) - real(s1) * B(1, 0);
        real_t scale1 = sqrt(abs(w1)) * sqrt(abs(w2));
        if (scale1 >= safmin and scale1 <= safmax) {
            w1 = w1 / scale1;
            w2 = w2 / scale1;
        }
        // Solve linear system
        w2 = w2 / B(1, 1);
        w1 = (w1 - B(0, 1) * w2) / B(0, 0);
        real_t scale2 = sqrt(abs(w1)) * sqrt(abs(w2));
        if (scale2 >= safmin and scale2 <= safmax) {
            w1 = w1 / scale2;
            w2 = w2 / scale2;
        }
        // Apply second shift
        v[0] = beta2 * (A(0, 0) * w1 + A(0, 1) * w2) -
               real(s2) * (B(0, 0) * w1 + B(0, 1) * w2);
        v[1] = beta2 * (A(1, 0) * w1 + A(1, 1) * w2) -
               real(s2) * (B(1, 0) * w1 + B(1, 1) * w2);
        v[2] = beta2 * (A(2, 0) * w1 + A(2, 1) * w2) -
               real(s2) * (B(2, 0) * w1 + B(2, 1) * w2);
        // Account for imaginary part
        v[0] = v[0] + imag(s1) * imag(s1) * B(0, 0) / scale1 / scale2;
        // Check for overflow
        if (abs(v[0]) > safmax or abs(v[1]) > safmax or abs(v[2]) > safmax) {
            v[0] = zero;
            v[1] = zero;
            v[2] = zero;
        }
    }
    return 0;
}

/** Given a 2-by-2 or 3-by-3 matrix pencil (A,B), lahqz_shiftcolumn
 *  calculates a multiple of the product:
 *  (beta2*A - s2*B)*B^(-1)*(beta1*A - s1*B)*B^(-1)*e1
 *
 *  This is used to introduce shifts in the QZ algorithm
 *
 * @return  0 if success
 *
 * @param[in] A 2x2 or 3x3 matrix.
 *      The matrix A as in the formula above.
 * @param[in] B 2x2 or 3x3 matrix.
 *      The matrix B as in the formula above.
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
int lahqz_shiftcolumn(const matrix_t& A,
                      const matrix_t& B,
                      vector_t& v,
                      complex_type<type_t<matrix_t>> s1,
                      complex_type<type_t<matrix_t>> s2,
                      real_type<type_t<matrix_t>> beta1,
                      real_type<type_t<matrix_t>> beta2)
{
    // Using
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // Constants
    const idx_t n = ncols(A);
    const T zero(0);
    const real_t safmin = safe_min<real_t>();
    const real_t safmax = safe_max<real_t>();

    // Check arguments
    tlapack_check_false((n != 2 and n != 3));
    tlapack_check_false(n != nrows(A));
    tlapack_check_false((idx_t)size(v) != n);

    if (n == 2) {
        T w1 = beta1 * A(0, 0) - s1 * B(0, 0);
        T w2 = beta1 * A(1, 0) - s1 * B(1, 0);
        v[0] = w1;
        v[1] = w2;
    }
    else {
        // Calculate first shifted vector
        T w1 = beta1 * A(0, 0) - s1 * B(0, 0);
        T w2 = beta1 * A(1, 0) - s1 * B(1, 0);
        real_t scale1 = sqrt(abs1(w1)) * sqrt(abs1(w2));
        if (scale1 >= safmin and scale1 <= safmax) {
            w1 = w1 / scale1;
            w2 = w2 / scale1;
        }
        // Solve linear system
        w2 = w2 / B(1, 1);
        w1 = (w1 - B(0, 1) * w2) / B(0, 0);
        real_t scale2 = sqrt(abs1(w1)) * sqrt(abs1(w2));
        if (scale2 >= safmin and scale2 <= safmax) {
            w1 = w1 / scale2;
            w2 = w2 / scale2;
        }
        // Apply second shift
        v[0] = beta2 * (A(0, 0) * w1 + A(0, 1) * w2) -
               s2 * (B(0, 0) * w1 + B(0, 1) * w2);
        v[1] = beta2 * (A(1, 0) * w1 + A(1, 1) * w2) -
               s2 * (B(1, 0) * w1 + B(1, 1) * w2);
        v[2] = beta2 * (A(2, 0) * w1 + A(2, 1) * w2) -
               s2 * (B(2, 0) * w1 + B(2, 1) * w2);
        // Check for overflow
        if (abs1(v[0]) > safmax or abs1(v[1]) > safmax or abs1(v[2]) > safmax) {
            v[0] = zero;
            v[1] = zero;
            v[2] = zero;
        }
    }
    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LAHQZ_SHIFTCOLUMN_HH
