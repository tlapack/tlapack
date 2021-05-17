#ifndef __TLAPACK_HH__
#define __TLAPACK_HH__

// BLAS

#include "tblas.hpp"

// Optimized LAPACK

#ifdef USE_LAPACKPP_WRAPPERS

    #ifndef LAPACK_UTIL_HH
        #define LAPACK_UTIL_HH // So as not to include utils from lapack++
    #endif

    #include "lapack/config.h"
    #include "lapack/types.hpp"
    #include "lapack/wrappers.hh" // from lapack++

#endif

// Template LAPACK

#include "lapack/lassq.hpp"

#endif // __TLAPACK_HH__
