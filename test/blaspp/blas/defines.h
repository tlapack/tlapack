// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEFINES_H
#define BLAS_DEFINES_H

// Using defines.h from BLAS++ to enforce the integer types of <T>BLAS
#define BLAS_SIZE_T std::int64_t
#define BLAS_INT_T  std::size_t

#define BLAS_ERROR_NDEBUG // Don't test corner cases

#endif        //  #ifndef BLAS_DEFINES_H