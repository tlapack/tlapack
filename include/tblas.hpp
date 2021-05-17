#ifndef __TLAPACK_BLAS_HH__
#define __TLAPACK_BLAS_HH__

// Optimized BLAS

#if defined(USE_BLASPP_WRAPPERS) && !defined(USE_BLASPP_TEMPLATES)

    // #if OPTIMIZED_BLAS == MKLBLAS
    //     #include "blas/mkl/blas.hpp"
    // #elif OPTIMIZED_BLAS == ONEMKLBLAS
    //     #include "blas/onemkl/blas.hpp"
    // #elif OPTIMIZED_BLAS == OPENBLAS
    //     #include "blas/openblas/blas.hpp"
    // #endif

    #ifndef BLAS_UTIL_HH
        #define BLAS_UTIL_HH // So as not to include utils from blas++
    #endif
    
    #include "blas/types.hpp"
    #include "blas/wrappers.hh" // from blas++

#endif

// Template BLAS

#ifdef USE_BLASPP_TEMPLATES

    #include <cstdint> // Contains std::int64_t
    #include <cmath> // Contains std::abs

    namespace blas {
        using size_t = std::int64_t;
        using int_t  = std::int64_t;
        const blas::size_t INVALID_INDEX = -1;

        // -----------------------------------------------------------------------------
        // is nan
        template< typename T >
        inline bool isnan( T x )
        {
            return x != x;
        }
    }

    #include "blas.hh"

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