// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_CONFIG_HH__
#define __TBLAS_CONFIG_HH__

#include "blas/defines.h"

// -----------------------------------------------------------------------------
// Integer types BLAS_SIZE_T and BLAS_INT_T

#ifndef BLAS_SIZE_T
    #define BLAS_SIZE_T size_t
#endif

#ifndef BLAS_INT_T
    #define BLAS_INT_T int64_t
#endif

#endif // __TBLAS_CONFIG_HH__