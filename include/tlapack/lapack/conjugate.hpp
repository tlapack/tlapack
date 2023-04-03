/// @file conjugate.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_CONJUGATE_HH
#define TLAPACK_CONJUGATE_HH

#include "tlapack/base/constants.hpp"
#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Conjugates a vector
 *
 * @param[in] x Vector of size n.
 *
 * @ingroup auxiliary
 */
template <class vector_t>
void conjugate(const vector_t& x)
{
    using idx_t = size_type<vector_t>;

    for (idx_t i = 0; i < size(x); ++i) {
        x[i] = conj(x[i]);
    }
}

}  // namespace tlapack

#endif  // TLAPACK_CONJUGATE_HH
