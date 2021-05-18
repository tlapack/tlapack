#ifndef __LASSQ_HH__
#define __LASSQ_HH__

#include "lapack/types.hpp"

namespace lapack {

template< typename TX >
void lassq(
    blas::size_t n,
    TX const* X, blas::int_t incx,
    real_type<TX> &scale,
    real_type<TX> &sumsq) {

    throw std::exception();  // not yet implemented
}

} // lapack

#endif // __LASSQ_HH__