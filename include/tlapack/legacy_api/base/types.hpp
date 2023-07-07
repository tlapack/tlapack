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

/** @def TLAPACK_SIZE_T
 * @brief Type of all size-related integers in libtlapack_c, libtlapack_cblas,
 * libtlapack_fortran, and in the routines of the legacy API.
 *
 * Supported types:
 *      int, short, long, long long, int8_t, int16_t,
 *      int32_t, int64_t, int_least8_t, int_least16_t,
 *      int_least32_t, int_least64_t, int_fast8_t,
 *      int_fast16_t, int_fast32_t, int_fast64_t,
 *      intmax_t, intptr_t, ptrdiff_t,
 *      size_t, uint8_t, uint16_t, uint32_t, uint64_t
 *
 * @note TLAPACK_SIZE_T must be std::int64_t if TLAPACK_USE_LAPACKPP is defined
 */
#ifdef TLAPACK_USE_LAPACKPP
    #ifndef TLAPACK_SIZE_T
        #define TLAPACK_SIZE_T std::int64_t
    #endif
#else
    #ifndef TLAPACK_SIZE_T
        #define TLAPACK_SIZE_T std::size_t
    #endif
#endif

/** @def TLAPACK_INT_T
 * @brief Type of all non size-related integers in libtlapack_c,
 * libtlapack_cblas, libtlapack_fortran, and in the routines of the legacy API.
 * It is the type used for the array increments, e.g., incx and incy.
 *
 * Supported types:
 *      int, short, long, long long, int8_t, int16_t,
 *      int32_t, int64_t, int_least8_t, int_least16_t,
 *      int_least32_t, int_least64_t, int_fast8_t,
 *      int_fast16_t, int_fast32_t, int_fast64_t,
 *      intmax_t, intptr_t, ptrdiff_t
 *
 * @note TLAPACK_INT_T must be std::int64_t if TLAPACK_USE_LAPACKPP is defined
 */
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
