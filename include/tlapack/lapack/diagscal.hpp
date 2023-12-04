#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Add scaled vector, $y := \alpha x + y$.
 *
 * @param[in] alpha Scalar.
 * @param[in] x     A n-element vector.
 * @param[in,out] y A vector with at least n elements.
 *
 * @ingroup blas1
 */
template <TLAPACK_MATRIX matrixA_t, TLAPACK_VECTOR vectorX_t>
void diagscal(const matrixA_t& A,
          const vectorX_t& x, int start_idx)
{
    using idx_t = size_type<vectorX_t>;
    using TA = type_t<matrixA_t>;
    using range = pair<idx_t, idx_t>;
    const idx_t m = nrows(A);
    // constants
    const idx_t n = size(x);
    const idx_t start = idx_t(start_idx); 
    for (int i = start; i < x; i++){
        auto v = slice(A, range{i, m}, i);
        auto norm = lange(tlapack::INF_NORM, v);
        x[i] *= float(norm);
        for (int j = 0; j < x; j++)
            A(i,j) /= sqrt(norm);
    }
}
}