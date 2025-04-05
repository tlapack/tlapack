/// @file pttrf.hpp Computes the Cholesky factorization of a Hermitian positive
/// definite tridiagonal matrix A.
/// @author Hugh M. Kadhem, University of California Berkeley, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_PTTRF_HH
#define TLAPACK_PTTRF_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Computes the Cholesky factorization of a Hermitian
 * positive definite tridiagonal matrix A.
 *
 * The factorization has the form
 *      $A = L D L^H$,
 * where L is unit lower bidiagonal and D is diagonal.
 *
 * @param[in,out] D
 *      On entry, the diagonal of $A$.
 *
 *      - On successful exit, the factor $D$.
 *
 * @param[in,out] E
 *      On entry, the first subdiagonal of $A$.
 *
 *      - On successful exit, the first subdiagonal of the factor $L$.
 *
 * @param[in] opts Options.
 *      Define the behavior of checks for NaNs.
 *
 * @return 0: successful exit.
 * @return i, 0 < i < n, if the leading minor of order i is not
 *      positive definite, and the factorization could not be completed.
 * @return n, if the leading minor of order n is not
 *      positive definite.
 *
 * @ingroup variant_interface
 */
template <class d_t,
          class e_t,
          enable_if_t<is_same_v<real_type<type_t<d_t>>, real_type<type_t<e_t>>>,
                      int> = 0>
int pttrf(d_t& D, e_t& E, const EcOpts& opts = {})
{
    using T = type_t<d_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<d_t>;
    // Constants
    const idx_t n = size(D);
    // check arguments
    tlapack_check(size(D) == size(E) + 1);
    // Quick return
    if (n <= 0) return 0;
    for (idx_t i = 0; i < n - 1; ++i) {
        if (real(D[i]) <= real_t(0)) {
            tlapack_error_if(
                opts.ec.internal, i + 1,
                "The leading minor of the reported order is not positive "
                "definite,"
                " and the factorization could not be completed.");
            return i + 1;
        }
        T Ei = E[i];
        E[i] = Ei / real(D[i]);
        D[i + 1] -= real(E[i] * conj(Ei));
    }
    if (real(D[n - 1]) <= real_t(0)) {
        tlapack_error_if(
            opts.ec.internal, n,
            "The leading minor of the highest order is not positive "
            "definite.");
        return n;
    }
    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_PTTRF_HH
