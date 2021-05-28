// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LASSQ_HH__
#define __LASSQ_HH__

#include "lapack/types.hpp"

namespace lapack {

/**
 * TODO: brief and detail
 * 
 * @param[in] n 
 * @param[in] x 
 * @param[in] incx 
 * @param[in,out] scale 
 * @param[in,out] sumsq 
 */
template< typename TX >
void lassq(
    blas::size_t n,
    TX const* x, blas::int_t incx,
    real_type<TX> &scale,
    real_type<TX> &sumsq) {

    throw std::exception();  // not yet implemented
}

} // lapack

#endif // __LASSQ_HH__