/// @file lamrg.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAMRG_HH
#define TLAPACK_LAMRG_HH

//
#include "tlapack/base/utils.hpp"

//

namespace tlapack {

/** Computes the eigenvalues of a real symmetric 2x2 matrix A
 *  [ a b ]
 *  [ b c ]
 *
 * @param[in] a
 *      Element (0,0) of A.
 * @param[in] b
 *      Element (0,1) and (1,0) of A.
 * @param[in] c
 *      Element (1,1) of A.
 * @param[out] s1
 *      The eigenvalue of A with the largest absolute value.
 * @param[out] s2
 *      The eigenvalue of A with the smallest absolute value.
 *
 * \verbatim
 *  s1 is accurate to a few ulps barring over/underflow.
 *
 *  s2 may be inaccurate if there is massive cancellation in the
 *  determinant a*c-b*b; higher precision or correctly rounded or
 *  correctly truncated arithmetic would be needed to compute s2
 *  accurately in all cases.
 *
 *  Overflow is possible only if s1 is within a factor of 5 of overflow.
 *  Underflow is harmless if the input data is 0 or exceeds
 *     underflow_threshold / macheps.
 * \endverbatim
 *
 *
 * @ingroup auxiliary
 */
template <class real_t, class idx_t, class a_t, class index_t>
void lamrg(
    idx_t n1, idx_t n2, a_t& a, real_t dtrd1, real_t dtrd2, index_t& index)
{
    idx_t n1sv = n1;
    idx_t n2sv = n2;
    idx_t ind1;
    idx_t ind2;

    if (dtrd1 > 0) {
        ind1 = 0;
    }
    else {
        ind1 = n1 - 1;
    }

    if (dtrd2 > 0) {
        ind2 = n1;
    }
    else {
        ind2 = n1 + n2 - 1;
    }

    idx_t i = 0;

    while (n1sv > 0 && n2sv > 0) {
        if (a[ind1] <= a[ind2]) {
            index[i] = ind1;
            i = i + 1;
            ind1 = ind1 + dtrd1;
            n1sv = n1sv - 1;
        }
        else {
            index[i] = ind2;
            i = i + 1;
            ind2 = ind2 + dtrd2;
            n2sv = n2sv - 1;
        }
    }

    if (n1sv == 0) {
        for (idx_t j = 0; j < n2sv; j++) {
            index[i] = ind2;
            i = i + 1;
            ind2 = ind2 + dtrd2;
        }
    }
    else {
        for (idx_t j = 0; j < n1sv; j++) {
            index[i] = ind1;
            i = i + 1;
            ind1 = ind1 + dtrd1;
        }
    }

    return;
}
}  // namespace tlapack

#endif  // TLAPACK_LAMRG_HH