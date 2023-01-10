// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEFINES_H
#define BLAS_DEFINES_H

// Using defines.h from BLAS++ to enforce the integer types of <T>LAPACK
#define TLAPACK_SIZE_T std::int64_t
#define TLAPACK_INT_T  std::int64_t

// Do not test corner cases since <T>LAPACK does not match
// BLAS++ in this matter
#undef assert_throw
#define assert_throw( expr, exception_type ) ((void)0)

#undef USE_LAPACKPP_WRAPPERS // Disable BLAS++ wrappers
#define TLAPACK_ERROR_NDEBUG // Don't test corner cases

#endif        //  #ifndef BLAS_DEFINES_H