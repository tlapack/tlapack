/// @file legacy_api/blas/nrm2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
///
/// Anderson E. (2017)
/// Algorithm 978: Safe Scaling in the Level 1 BLAS
/// ACM Trans Math Softw 44:1--28
/// @see https://doi.org/10.1145/3061665
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_NRM2_HH
#define TLAPACK_LEGACY_NRM2_HH

#include "tlapack/blas/nrm2.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {
namespace legacy {

    /**
     * @return 2-norm of vector,
     *     $|| x ||_2 = (\sum_{i=0}^{n-1} |x_i|^2)^{1/2}$.
     *
     * Generic implementation for arbitrary data types.
     *
     * @param[in] n
     *     Number of elements in x. n >= 0.
     *
     * @param[in] x
     *     The n-element vector x, in an array of length (n-1)*incx + 1.
     *
     * @param[in] incx
     *     Stride between elements of x. incx > 0.
     *
     * @ingroup legacy_blas
     */
    template <typename T>
    real_type<T> nrm2(idx_t n, T const* x, int_t incx)
    {
        tlapack_check_false(incx <= 0);

        // quick return
        if (n <= 0) return 0;

        tlapack_expr_with_vector_positiveInc(x_, T, n, x, incx,
                                             return nrm2(x_));
    }

}  // namespace legacy
}  // namespace tlapack

#endif  // #ifndef TLAPACK_LEGACY_NRM2_HH
