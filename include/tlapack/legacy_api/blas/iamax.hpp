/// @file legacy_api/blas/iamax.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_IAMAX_HH
#define TLAPACK_LEGACY_IAMAX_HH

#include "tlapack/blas/iamax.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {
namespace legacy {

    /**
     * @brief Return $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$
     *
     * @param[in] n
     *     Number of elements in x.
     *
     * @param[in] x
     *     The n-element vector x, in an array of length (n-1)*incx + 1.
     *
     * @param[in] incx
     *     Stride between elements of x. incx > 0.
     *
     * @return In priority order:
     * 1. 0 if n <= 0,
     * 2. the index of the first `NAN` in $x$ if it exists and if `checkNAN ==
     * true`,
     * 3. the index of the first `Infinity` in $x$ if it exists,
     * 4. the Index of the infinity-norm of $x$, $|| x ||_{inf}$,
     *     $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$.
     *
     * @see iamax( const vector_t& x, const ec_opts_t& opts = {} )
     *
     * @ingroup legacy_blas
     */
    template <typename T>
    idx_t iamax(idx_t n, T const* x, int_t incx)
    {
        tlapack_check_false(incx <= 0);

        // quick return
        if (n <= 0) return 0;

        tlapack_expr_with_vector_positiveInc(x_, T, n, x, incx,
                                             return iamax(x_));
    }

}  // namespace legacy
}  // namespace tlapack

#endif  //  #ifndef TLAPACK_LEGACY_IAMAX_HH
