#ifndef __TLAPACK_BLAS_HH__
#define __TLAPACK_BLAS_HH__

// // Definitions:
#include "defines.hpp"
// #include "types.hpp"

// // Optimized BLAS
// #ifdef OPTIMIZED_BLAS
//     #if OPTIMIZED_BLAS == MKLBLAS
//         #include "blas/mkl/blas.hpp"
//     #elif OPTIMIZED_BLAS == ONEMKLBLAS
//         #include "blas/onemkl/blas.hpp"
//     #elif OPTIMIZED_BLAS == OPENBLAS
//         #include "blas/openblas/blas.hpp"
//     #endif
// #endif

// BLASPP
#ifdef USE_BLASPP
    // #include "blas.hh"
    // =============================================================================
    // Level 1 BLAS template implementations

    #include "blas/asum.hh"
    #include "blas/axpy.hh"
    #include "blas/copy.hh"
    #include "blas/dot.hh"
    #include "blas/dotu.hh"
    #include "blas/iamax.hh"
    #include "blas/nrm2.hh"
    #include "blas/rot.hh"
    #include "blas/rotg.hh"
    #include "blas/rotm.hh"
    #include "blas/rotmg.hh"
    #include "blas/scal.hh"
    #include "blas/swap.hh"

    // =============================================================================
    // Level 2 BLAS template implementations

    #include "blas/gemv.hh"
    #include "blas/ger.hh"
    #include "blas/geru.hh"
    #include "blas/hemv.hh"
    #include "blas/her.hh"
    #include "blas/her2.hh"
    #include "blas/symv.hh"
    #include "blas/syr.hh"
    #include "blas/syr2.hh"
    #include "blas/trmv.hh"
    #include "blas/trsv.hh"

    // =============================================================================
    // Level 3 BLAS template implementations

    #include "blas/gemm.hh"
    #include "blas/hemm.hh"
    #include "blas/herk.hh"
    #include "blas/her2k.hh"
    #include "blas/symm.hh"
    #include "blas/syrk.hh"
    #include "blas/syr2k.hh"
    #include "blas/trmm.hh"
    #include "blas/trsm.hh"

    // Other BLAS templates
    #include "blas/gbmv.hpp"
    // #include "blas/sbmv.hpp"
    // #include "blas/spmv.hpp"
    // #include "blas/spr.hpp"
    // #include "blas/spr2.hpp"
    // #include "blas/tbmv.hpp"
    // #include "blas/tpmv.hpp"
    // #include "blas/tbsv.hpp"
    // #include "blas/tpsv.hpp"

#else

    // =============================================================================
    // Level 1 BLAS template implementations

    #include "blas/asum.hh"
    #include "blas/axpy.hh"
    #include "blas/copy.hh"
    #include "blas/dot.hh"
    #include "blas/dotu.hh"
    #include "blas/iamax.hh"
    #include "blas/nrm2.hh"
    #include "blas/rot.hh"
    #include "blas/rotg.hh"
    #include "blas/rotm.hh"
    #include "blas/rotmg.hh"
    #include "blas/scal.hh"
    #include "blas/swap.hh"

    // =============================================================================
    // Level 2 BLAS template implementations

    #include "blas/gemv.hh"
    #include "blas/ger.hh"
    #include "blas/geru.hh"
    #include "blas/gbmv.hpp"
    #include "blas/hemv.hh"
    #include "blas/her.hh"
    #include "blas/her2.hh"
    #include "blas/symv.hh"
    #include "blas/syr.hh"
    #include "blas/syr2.hh"
    #include "blas/spmv.hpp"
    #include "blas/spr.hpp"
    #include "blas/spr2.hpp"
    #include "blas/sbmv.hpp"
    #include "blas/trmv.hh"
    #include "blas/trsv.hh"
    #include "blas/tpmv.hpp"
    #include "blas/tbmv.hpp"
    #include "blas/tpsv.hpp"
    #include "blas/tbsv.hpp"

    // =============================================================================
    // Level 3 BLAS template implementations

    #include "blas/gemm.hh"
    #include "blas/hemm.hh"
    #include "blas/herk.hh"
    #include "blas/her2k.hh"
    #include "blas/symm.hh"
    #include "blas/syrk.hh"
    #include "blas/syr2k.hh"
    #include "blas/trmm.hh"
    #include "blas/trsm.hh"

#endif

#endif // __BLAS_HH__