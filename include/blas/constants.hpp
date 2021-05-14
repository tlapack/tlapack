#ifndef __TLAPACK_BLAS_CONSTANTS_HH__
#define __TLAPACK_BLAS_CONSTANTS_HH__

#include <type_traits>
#include <limits>
#include "blas/types.hpp"

namespace blas {

const blas::size_t INVALID_INDEX = ( std::is_unsigned<blas::size_t>::value )
    ? std::numeric_limits< blas::size_t >::max() // If unsigned, max value is an invalid index
    : -1;                                        // If signed, -1 is the default invalid index

}

#endif // __TLAPACK_BLAS_CONSTANTS_HH__