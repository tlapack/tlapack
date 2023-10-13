/// @file inv_house3.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_INV_HOUSE_HH
#define TLAPACK_INV_HOUSE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** @brief Inv_house calculates a reflector to reduce
 * a the first column in a 3x3 matrix from the right to a unit vector. Note that
 * this is special because a reflector applied from the right would usually
 * reduce a row, not a column. This is known as an inverse reflector.
 *
 * We need to solve the system
 * A x = e_1
 * If we then calculate a reflector that reduces x:
 * H x = alpha e_1,
 * then H is the reflector that we were looking for.
 *
 * Because the reflector is invariant w.r.t. the scale of x,
 * we will solve a scaled system so that x0 = 1.
 * The rest of x can then be calculated using:
 * [A11 A12] [x1] = - scale * [A10]
 * [A21 A22] [x2]             [A20]
 *
 * We solve this system of equations using fully pivoted LU
 *
 * @param[in] A 3x3 matrix.
 * @param[out] v vector of size 3
 * @param[out] tau
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
void inv_house3(const matrix_t& A, vector_t& v, type_t<vector_t>& tau)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    using idx_t = size_type<matrix_t>;

    const real_t safemin = safe_min<real_t>();

    T a11, a12, a21, a22;

    // Swap rows if necessary
    real_t temp1 = max<real_t>(abs1(A(1, 1)), abs1(A(1, 2)));
    real_t temp2 = max<real_t>(abs1(A(2, 1)), abs1(A(2, 2)));

    if (temp1 < safemin and temp2 < safemin) {
        v[0] = (T)0;
        v[1] = (T)1;
        v[2] = (T)0;
        larfg(FORWARD, COLUMNWISE_STORAGE, v, tau);
        return;
    }
    if (temp1 >= temp2) {
        a11 = A(1, 1);
        a12 = A(1, 2);
        a21 = A(2, 1);
        a22 = A(2, 2);
        v[1] = -A(1, 0);
        v[2] = -A(2, 0);
    }
    else {
        a11 = A(2, 1);
        a12 = A(2, 2);
        a21 = A(1, 1);
        a22 = A(1, 2);
        v[1] = -A(2, 0);
        v[2] = -A(1, 0);
    }

    // Swap columns if necessary
    bool colswapped = false;
    if (abs1(a12) > abs1(a11)) {
        colswapped = true;
        std::swap(a11, a12);
        std::swap(a21, a22);
    }

    // Calculate LU factorization
    T u11 = a11;
    T u12 = a12;
    T l21 = a21 / u11;
    T u22 = a22 - l21 * u12;

    // Solve lower triangular system
    v[2] = v[2] - v[1] * l21;
    // Solve upper triangular system
    real_t scale = (real_t)1;
    if (abs1(u22) < safemin) {
        v[0] = (T)0;
        v[1] = (T)1;
        v[2] = -u12 / u11;
        larfg(FORWARD, COLUMNWISE_STORAGE, v, tau);
        return;
    }
    if (abs1(u22) < abs1(v[2])) scale = abs1(u22 / v[2]);
    if (abs1(u11) < abs1(v[1])) scale = min(scale, abs1(u11 / v[1]));
    v[2] = (scale * v[2]) / u22;
    v[1] = (scale * v[1] - u12 * v[2]) / u11;
    v[0] = scale;
    if (colswapped) std::swap(v[1], v[2]);

    // Calculate Reflector
    larfg(FORWARD, COLUMNWISE_STORAGE, v, tau);
}

}  // namespace tlapack

#endif  // TLAPACK_INV_HOUSE_HH
