#ifndef UNGL2
#define UNGL2

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
 * hand-crafted full LQ factorization for a general m-by-n column major matrix A (m <= n), level 0, right looking algorithm
 * outputing Q is an orthonormal matrix, allowing for bidiagonal reduction, n <= m, and the inclusion of m <= k <=n.
 * @param[in,out] Q
 */
template <typename matrix_t, class vector_t>
int ungl2(matrix_t &Q, vector_t &tauw, vector_t &work, bool &bidg)
{

    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = std::pair<idx_t, idx_t>;

    const idx_t k = nrows(Q);
    const idx_t n = ncols(Q);

    const idx_t m = size(tauw); // max number of reflectors to use
    const idx_t t = std::min(k, m);      // number of reflectors to use

    std::vector<T> C_(k * n);
    legacyMatrix<T> C(k, n, &C_[0], k);

    // Identity matrix C k-by-n to start with
    for (idx_t j = 0; j < n; ++j)
    {
        for (idx_t i = 0; i < k; ++i)
            C(i, j) = make_scalar<T>(0, 0);
        if (j < k)
        {
            C(j, j) = make_scalar<T>(1, 0);
        }
    }

    if (bidg == 1) //for bidiagonal reduction, part of brd
    {
        for (idx_t j = n - 2; j != -1; --j)
        {
            auto w = slice(Q, j, range(j + 1, n));
            auto Z11 = slice(C, range(0, n), range(j + 1, n));

            larf(Side::Right, w, conj(tauw[j]), Z11, work);
        }
    }
    else
    {
        for (idx_t j = t - 1; j != -1; --j)
        {
            auto w = slice(Q, j, range(j, n));
            for (idx_t i = 0; i < n - j; ++i) // for loop to conj w.
                w[i] = conj(w[i]);

            auto C11 = slice(C, range(0, k), range(j, n));

            larf(Side::Right, w, conj(tauw[j]), C11, work);
            for (idx_t i = 0; i < n - j; ++i) // for loop to conj w back.
                w[i] = conj(w[i]);
        }
    }

    lacpy(Uplo::General, C, Q);

    return 0;
}
}
#endif // UNGL2