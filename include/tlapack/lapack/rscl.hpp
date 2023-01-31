/// @file rscl.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_RSCL_HH
#define TLAPACK_RSCL_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Scale vector by the reciprocal of a constant, $x := x / \alpha$.
 *
 * @param[in] alpha Scalar.
 * @param[in,out] x A n-element vector.
 *
 * @ingroup auxiliary
 */
template <class vector_t, class alpha_t, class T = type_t<vector_t>>
void rscl(const alpha_t& alpha, vector_t& x)
{
    using idx_t = size_type<vector_t>;
    const idx_t n = size(x);

    for (idx_t i = 0; i < n; ++i)
        x[i] /= alpha;
}

}  // namespace tlapack

#endif  // TLAPACK_RSCL_HH