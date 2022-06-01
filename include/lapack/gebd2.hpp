#ifndef GEBD2
#define GEBD2

// #include <utility>
// #include <plugins/tlapack_stdvector.hpp>
// #include <plugins/tlapack_legacyArray.hpp>
// #include <plugins/tlapack_debugutils.hpp>
// #include <tlapack.hpp>

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

/** 
* hand-crafted  reduction to bidiagonal form of a general m-by-n matrix A, level 2, right looking algorithm
* @param[in,out] A
*/

namespace tlapack {

template <typename matrix_t, class vector_t>
int gebd2( matrix_t & A, vector_t & tauv, vector_t & tauw, vector_t & work ){

    using idx_t = size_type< matrix_t >;
    using T = type_t< matrix_t >;
    using real_t = real_type<T>;
    using range = std::pair<idx_t, idx_t>;


    const idx_t m = nrows( A );
    const idx_t n = ncols( A );

    for (idx_t j = 0; j < n; ++j) {

        auto v = slice(A, range(j, m), j);
        
        larfg(v, tauv[j]); //gen the vertical reflector v's

        if( j < n-1 ){ 
            auto A11 = slice(A, range(j, m), range(j+1, n));
            larf(Side::Left, v, conj(tauv[j]), A11, work);
        }

        if( j < n-1 ){
            auto w = slice(A, j, range(j+1, n)); 
            //for loop to conj w.
            for (idx_t i = 0; i < n-j-1; ++i)
                w[i] = conj(w[i]);

            larfg(w, tauw[j]); //gen the horizontal reflector w's

            if( j < m-1 ){ 
                auto B11 = slice(A, range(j+1, m), range(j+1, n));
                larf(Side::Right, w, tauw[j], B11, work);
            }
            // for (idx_t i = 0; i < n-j-1; ++i) //for loop to conj w back.
            //     w[i] = conj(w[i]);
        }
    }

    return 0;
}
}
#endif //GEBD2