#ifndef __TLAPACK_HH__
#define __TLAPACK_HH__

// Definitions

#include "defines.hpp"

// BLAS

#include "blas.hpp"

// Optimized LAPACK

#ifdef USE_LAPACKPP_WRAPPERS

    /// Use to silence compiler warning of unused variable.
    #define blas_unused( var ) ((void)var)

    #include "lapack/wrappers.hh"

#endif

// Template LAPACK

#endif // __TLAPACK_HH__