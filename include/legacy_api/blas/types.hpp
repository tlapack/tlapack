// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// -----------------------------------------------------------------------------
// Integer types BLAS_SIZE_T and BLAS_INT_T

#include "blas/types.hpp"
#include <cstdint> // Defines std::int64_t
#include <cstddef> // Defines std::size_t

#if defined(USE_BLASPP_WRAPPERS) || defined(USE_LAPACKPP_WRAPPERS)
    #ifndef BLAS_SIZE_T
        #define BLAS_SIZE_T std::int64_t
    #endif
#else
    #ifndef BLAS_SIZE_T
        #define BLAS_SIZE_T std::size_t
    #endif
#endif

#ifndef BLAS_INT_T
    #define BLAS_INT_T std::int64_t
#endif
// -----------------------------------------------------------------------------

namespace blas {
    using idx_t = BLAS_SIZE_T;
    using int_t = BLAS_INT_T;
}