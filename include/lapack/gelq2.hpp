#ifndef GELQ2
#define GELQ2

// #include <utility>
#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>
// #include <plugins/tlapack_debugutils.hpp>
// #include <tlapack.hpp>

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace tlapack {

/** 
* hand-crafted LQ factorization for a general m-by-n row major matrix A (m < n), level 2, right looking algorithm
* outputing A in terms of L and reflector w's
* @param[in,out] A
*/

template <typename matrix_t, class vector_t, class work_t>
int gelq2(matrix_t & A, vector_t & tauw, work_t & work ){
    using idx_t = size_type< matrix_t >;
    using T = type_t< matrix_t >;
    using real_t = real_type<T>;
    using range = std::pair<idx_t, idx_t>;


    const idx_t m = nrows( A );
    const idx_t n = ncols( A );

    const idx_t k = std::min(m, n);

    for (idx_t j = 0; j < k; ++j) {
        auto w = slice(A, j, range(j, n)); 
        //for loop to conj w.
        for (idx_t i = 0; i < n-j; ++i)
            w[i] = conj(w[i]);

        larfg(w, tauw[j]); //gen the horizontal reflector w's

        if( j < k-1 || k < m){ 
            auto Q11 = slice(A, range(j+1, m), range(j, n));
            larf(Side::Right, w, tauw[j], Q11, work);
        }
        for (idx_t i = 0; i < n-j; ++i) //for loop to conj w back.
            w[i] = conj(w[i]);
    }

    return 0;
}
}


#endif //GELQ2