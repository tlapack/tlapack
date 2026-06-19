#include <tlapack/lapack/lange.hpp>

using namespace tlapack;

// Conducts check for tikhonov regularized least squares problem
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t,
          TLAPACK_MATRIX matrixx_t>
void tik_check(matrixA_t A,
               matrixb_t b,
               real_t lambda,
               matrixx_t x,
               real_t* normr,
               real_t* check_tikhonov)
{
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixb_t>;

    Create<matrixb_t> new_matrix;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    std::vector<T> y_;
    auto y = new_matrix(y_, n, k);

    std::vector<T> r_;
    auto r = new_matrix(r_, m, k);

    lacpy(GENERAL, b, r);

    // Compute b - A *x -> b
    gemm(NO_TRANS, NO_TRANS, real_t(-1), A, x, real_t(1), r);
    (*normr) = lange(FROB_NORM, r);

    // Compute A.H*(b - A x) -> y
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), A, r, y);

    // Compute A.H*(b - A x) - (lambda^2)*x -> y
    for (idx_t j = 0; j < k; j++)
        for (idx_t i = 0; i < n; i++)
            y(i, j) -= (lambda) * (lambda)*x(i, j);

    (*check_tikhonov) = lange(FROB_NORM, y);
    real_t normA = lange(FROB_NORM, A);

    (*check_tikhonov) = (*check_tikhonov) / normA;
}