#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>
#ifdef USE_MPFR
    #include <tlapack/plugins/mpreal.hpp>
#endif

// <T>LAPACK
#include <tlapack/blas/trsm.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include "../../eigen/Eigen/Core"
#include "tlapack/base/utils.hpp"

// C++ headers
#include <iostream>
#include <vector>

template<class T>
void rescale(matrix_t& A, matrix_t& S){
     auto A00 = tlapack::slice(A, range(0, k0), range(0, k0));
     auto A01 = tlapack::slice(A, range(0, k0), range(k0, n));
     auto A10 = tlapack::slice(A, range(k0, m), range(0, k0));
     auto A11 = tlapack::slice(A, range(k0, m), range(k0, n));


}