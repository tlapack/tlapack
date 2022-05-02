// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_BLAS_ROT_HH__
#define __TLAPACK_BLAS_ROT_HH__

#include "base/utils.hpp"

namespace tlapack {

/**
 * Apply plane rotation:
 * \[
 *       \begin{bmatrix} x^T   \\ y^T    \end{bmatrix}
 *     := \begin{bmatrix} c & s \\ -s & c \end{bmatrix}
 *       \begin{bmatrix} x^T   \\ y^T    \end{bmatrix}.
 * \]
 *
 * @see rotg to generate the rotation.
 *
 * @param[in,out] x A n-element vector.
 * @param[in,out] y A n-element vector.
 * @param[in] c     Cosine of rotation; real.
 * @param[in] s     Sine of rotation; scalar.
 *
 * @ingroup rot
 */
template<
    class vectorX_t, class vectorY_t,
    class c_type, class s_type
>
void rot(
    vectorX_t& x, vectorY_t& y,
    const c_type& c, const s_type& s )
{
    using idx_t = size_type< vectorX_t >;

    // constants
    const idx_t n = size(x);

    // check arguments
    tblas_error_if( size(y) != n );

    // quick return
    if ( n == 0 || (c == 1 && s == s_type(0)) )
        return;

    for (idx_t i = 0; i < n; ++i) {
        auto stmp = c*x[i] + s*y[i];
        y[i] = c*y[i] - conj(s)*x[i];
        x[i] = stmp;
    }
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_BLAS_ROT_HH__
