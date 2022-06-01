#ifndef GELQF
#define GELQF

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
 * hand-crafted general LQ factorization, level 3 BLAS, right looking algorithm
 * @param[in,out] A
 */

template <typename matrix_t, class vector_t, class work_t>
int gelqf(matrix_t &A, matrix_t &TT, vector_t &tauw, work_t &work, const size_type<matrix_t> &nb)
{

    // type alias for indexes
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = std::pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t j = 0; j < m; j += nb)
    {
        idx_t ib = std::min<idx_t>(nb, m - j);

        auto TT1 = slice(TT, range(j, j + ib), range(0, ib));
        auto A11 = slice(A, range(j, j + ib), range(j, n));

        auto tauw1 = slice(tauw, range(j, j + ib));

        gelq2(A11, tauw1, work);

        larft(Direction::Forward, 
            StoreV::Rowwise, A11, tauw1, TT1);

        if( j+ib < m ){                 
            auto A12 = slice(A, range(j+ib, m), range(j, n));
            auto work1 = slice(TT, range(j+ib, m), range(0, ib));

            larfb(
                    Side::Right, 
                    Op::NoTrans,
                    Direction::Forward, 
                    StoreV::Rowwise, 
                    A11, TT1, A12, work1);
        }
    }

    return 0;
}
}
#endif // GELQF