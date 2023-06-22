/// @file legacy_api/lapack/larfg.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/larfg.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LARFG_HH
#define TLAPACK_LEGACY_LARFG_HH

#include "tlapack/lapack/larfg.hpp"

namespace tlapack {
namespace legacy {

    template <typename T>
    void inline larfg(idx_t n, T& alpha, T* x, int_t incx, T& tau)
    {
        tlapack_expr_with_vector(
            x_, T, n - 1, x, incx,
            return larfg(columnwise_storage, alpha, x_, tau));
    }

    /** Generates a elementary Householder reflection.
     *
     * @see larfg( idx_t, T &, T *, int_t, T & )
     *
     * @ingroup legacy_lapack
     */
    template <typename T>
    void inline larfg(idx_t n, T* alpha, T* x, int_t incx, T* tau)
    {
        larfg(n, *alpha, x, incx, *tau);
    }

}  // namespace legacy
}  // namespace tlapack

#endif  // TLAPACK_LEGACY_LARFG_HH
