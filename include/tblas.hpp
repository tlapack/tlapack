// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of T-LAPACK.
// T-LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_HH__
#define __TBLAS_HH__

// -----------------------------------------------------------------------------
#include "defines.h"

// Optimized BLAS

#if defined(USE_BLASPP_WRAPPERS) && !defined(USE_BLASPP_TEMPLATES)

    #ifndef BLAS_UTIL_HH
        #define BLAS_UTIL_HH // So as not to include utils from blas++
    #endif
    
    #include "blas/types.hpp"
    #include "blas/wrappers.hh" // from blas++

#endif

// Template BLAS

#ifdef USE_BLASPP_TEMPLATES // If we use the templates from BLAS++

    #include <cstdint> // Contains std::int64_t
    #include <cmath> // Contains std::abs
    #include <limits> // Contains std::numeric_limits
    #include <complex> // Contains std::complex

    namespace blas {

        using size_t = std::int64_t;
        using int_t  = std::int64_t;
        const blas::size_t INVALID_INDEX = -1;

        // -----------------------------------------------------------------------------
        // Use routines from std C++
        using std::abs; // Contains the 2-norm for the complex case
        using std::isinf;
        using std::isnan;
        using std::ceil;
        using std::floor;
        using std::pow;
        using std::sqrt;

        // -----------------------------------------------------------------------------
        /// isnan for complex numbers
        template< typename T >
        inline bool isnan( const std::complex<T>& x )
        {
            return isnan( real(x) ) || isnan( imag(x) );
        }

        /** Blue's min constant b for the sum of squares
         * @see https://doi.org/10.1145/355769.355771
         * @ingroup utils
         */
        template <typename real_t>
        inline const real_t blue_min()
        {
            const real_t half( 0.5 );
            const int fradix = std::numeric_limits<real_t>::radix;
            const int expm   = std::numeric_limits<real_t>::min_exponent;

            return pow( fradix, ceil( half*(expm-1) ) );
        }

        /** Blue's max constant B for the sum of squares
         * @see https://doi.org/10.1145/355769.355771
         * @ingroup utils
         */
        template <typename real_t>
        inline const real_t blue_max()
        {
            const real_t half( 0.5 );
            const int fradix = std::numeric_limits<real_t>::radix;
            const int expM   = std::numeric_limits<real_t>::max_exponent;
            const int t      = std::numeric_limits<real_t>::digits;

            return pow( fradix, floor( half*( expM - t + 1 ) ) );
        }

        /** Blue's scaling constant for numbers smaller than b
         * 
         * @details Modification introduced in @see https://doi.org/10.1145/3061665
         *          to scale denormalized numbers correctly.
         * 
         * @ingroup utils
         */
        template <typename real_t>
        inline const real_t blue_scalingMin()
        {
            const real_t half( 0.5 );
            const int fradix = std::numeric_limits<real_t>::radix;
            const int expm   = std::numeric_limits<real_t>::min_exponent;
            const int t      = std::numeric_limits<real_t>::digits;

            return pow( fradix, -floor( half*(expm-t) ) );
        }

        /** Blue's scaling constant for numbers bigger than B
         * @see https://doi.org/10.1145/355769.355771
         * @ingroup utils
         */
        template <typename real_t>
        inline const real_t blue_scalingMax()
        {
            const real_t half( 0.5 );
            const int fradix = std::numeric_limits<real_t>::radix;
            const int expM   = std::numeric_limits<real_t>::max_exponent;
            const int t      = std::numeric_limits<real_t>::digits;

            return pow( fradix, -ceil( half*( expM + t - 1 ) ) );
        }
    }

    #include "blas.hh" // Include BLAS++

#else

    // =============================================================================
    // Level 1 BLAS template implementations

    #include "blas/asum.hpp"
    #include "blas/axpy.hpp"
    #include "blas/copy.hpp"
    #include "blas/dot.hpp"
    #include "blas/dotu.hpp"
    #include "blas/iamax.hpp"
    #include "blas/nrm2.hpp"
    #include "blas/rot.hpp"
    #include "blas/rotg.hpp"
    #include "blas/rotm.hpp"
    #include "blas/rotmg.hpp"
    #include "blas/scal.hpp"
    #include "blas/swap.hpp"

    // =============================================================================
    // Level 2 BLAS template implementations

    #include "blas/gemv.hpp"
    #include "blas/ger.hpp"
    #include "blas/geru.hpp"
    #include "blas/gbmv.hpp"
    #include "blas/hemv.hpp"
    #include "blas/her.hpp"
    #include "blas/her2.hpp"
    #include "blas/symv.hpp"
    #include "blas/syr.hpp"
    #include "blas/syr2.hpp"
    // #include "blas/spmv.hpp"
    // #include "blas/spr.hpp"
    // #include "blas/spr2.hpp"
    // #include "blas/sbmv.hpp"
    #include "blas/trmv.hpp"
    #include "blas/trsv.hpp"
    // #include "blas/tpmv.hpp"
    // #include "blas/tbmv.hpp"
    // #include "blas/tpsv.hpp"
    // #include "blas/tbsv.hpp"

    // =============================================================================
    // Level 3 BLAS template implementations

    #include "blas/gemm.hpp"
    #include "blas/hemm.hpp"
    #include "blas/herk.hpp"
    #include "blas/her2k.hpp"
    #include "blas/symm.hpp"
    #include "blas/syrk.hpp"
    #include "blas/syr2k.hpp"
    #include "blas/trmm.hpp"
    #include "blas/trsm.hpp"

#endif

#endif // __BLAS_HH__