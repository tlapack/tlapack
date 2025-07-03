/// @file lamrg.hpp
/// @author Brian Dang, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAMRG_HH
#define TLAPACK_LAMRG_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** DLAMRG creates a permutation list to merge the entries of two independently
 *sorted sets into a single set sorted in ascending order.
 *
 * \verbatim
 *      DLAMRG will create a permutation list which will merge the elements
 *      of A (which is composed of two independently sorted sets) into a
 *      single set which is sorted in ascending order.
 * \endverbatim
 *
 * @param[in] n1
 *      N2 is INTEGER
 * @param[in] n2
 *      N2 is INTEGER
 *      These arguments contain the respective lengths of the two
 *      sorted lists to be merged.
 * @param[in] a
 *      A is DOUBLE PRECISION array, dimension (N1+N2)
 *      The first N1 elements of A contain a list of numbers which
 *      are sorted in either ascending or descending order. Likewise
 *      for the final N2 elements.
 * @param[in] dtrd1
 *      DTRD1 is INTEGER
 * @param[in] dtrd2
 *      DTRD2 is INTEGER
 *      These are the strides to be taken through the array A.
 *      Allowable strides are 1 and -1.  They indicate whether a
 *      subset of a is sorted in ascending (DTRDx = 1) or descending
 *      (DTRDx = -1) order.
 * @param[out] index
 *      INDEX is INTEGER array, dimension (N1+N2)
 *      On exit this array will contain a permutation such that
 *      if B( I ) = A( INDEX( I ) ) for I=1,N1+N2, then B will be
 *      sorted in ascending order.
 *
 * \verbatim
 *       INDEX is INTEGER array, dimension (N1+N2)
 *       On exit this array will contain a permutation such that
 *       if B( I ) = A( INDEX( I ) ) for I=1,N1+N2, then B will be
 *       sorted in ascending order.
 * \endverbatim
 *
 * @ingroup lamrg
 */
template <class idx_t, class a_t, class idx1_t>
void lamrg(idx_t n1, idx_t n2, a_t& a, int dtrd1, int dtrd2, idx1_t& index)
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