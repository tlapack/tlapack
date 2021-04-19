#ifndef __TLAPACK_TYPES_HH__
#define __TLAPACK_TYPES_HH__

#include <vector>
#include <complex>
#include <cstdint>

namespace tlapack {

    // -----------------------------------------------------------------------------
    using size_t = int64_t;
    using int_t  = int64_t;

    // -----------------------------------------------------------------------------
    typedef enum { ColMajor = 'C', RowMajor = 'R' } Layout;
    typedef enum { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' } Op;
    typedef enum { Upper    = 'U', Lower    = 'L', General   = 'G' } Uplo;
    typedef enum { NonUnit  = 'N', Unit     = 'U' } Diag;
    typedef enum { Left     = 'L', Right    = 'R' } Side;

}

#endif // __TLAPACK_TYPES_HH__