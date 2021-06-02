// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of T-LAPACK.
// T-LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_CONSTANTS_HH__
#define __TBLAS_CONSTANTS_HH__

#include <type_traits>
#include <limits>
#include "blas/types.hpp"

namespace blas {

/// INVALID_INDEX = ( std::is_unsigned<blas::size_t>::value )
///  ? std::numeric_limits< blas::size_t >::max() // If unsigned, max value is an invalid index
///  : -1;                                        // If signed, -1 is the default invalid index
const blas::size_t INVALID_INDEX( -1 );

}

#endif // __TBLAS_CONSTANTS_HH__