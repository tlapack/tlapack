/// @file legacy_api/base/types.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_TYPES_HH
#define TLAPACK_LEGACY_TYPES_HH

#include "tlapack/base/types.hpp"

// -----------------------------------------------------------------------------
// Integer types TLAPACK_SIZE_T and TLAPACK_INT_T

#include <cstddef>  // Defines std::size_t
#include <cstdint>  // Defines std::int64_t

#ifdef USE_LAPACKPP_WRAPPERS
    #ifndef TLAPACK_SIZE_T
        #define TLAPACK_SIZE_T std::int64_t
    #endif
#else
    #ifndef TLAPACK_SIZE_T
        #define TLAPACK_SIZE_T std::size_t
    #endif
#endif

#ifndef TLAPACK_INT_T
    #define TLAPACK_INT_T std::int64_t
#endif
// -----------------------------------------------------------------------------

namespace tlapack {
namespace legacy {

    using idx_t = TLAPACK_SIZE_T;
    using int_t = TLAPACK_INT_T;

    // -----------------------------------------------------------------------------
    // lascl
    enum class MatrixType {
        General = 'G',
        Lower = 'L',
        Upper = 'U',
        Hessenberg = 'H',
        LowerBand = 'B',
        UpperBand = 'Q',
        Band = 'Z',
    };

}  // namespace legacy
}  // namespace tlapack

#endif  // TLAPACK_LEGACY_TYPES_HH
