#include <tlapack/lapack/lange.hpp>

using namespace tlapack;

// Conducts check for tikhonov regularized least squares problem
template <TLAPACK_MATRIX matrixA_copy_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t,
          TLAPACK_MATRIX matrixx_t>
void tik_check(matrixA_copy_t A_copy, matrixb_t b, real_t lambda, matrixx_t x)
{
    using T = type_t<matrixA_copy_t>;
    using idx_t = size_type<matrixb_t>;

    Create<matrixb_t> new_matrix;

    const idx_t n = ncols(A_copy);
    const idx_t k = ncols(b);

    std::vector<T> y_;
    auto y = new_matrix(y_, n, k);

    // Compute b - A *x -> b
    gemm(NO_TRANS, NO_TRANS, real_t(-1), A_copy, x, real_t(1), b);

    // Compute A.H*(b - A x) -> y
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), A_copy, b, y);

    // Compute A.H*(b - A x) - (lambda^2)*x -> y
    for (idx_t j = 0; j < k; j++)
        for (idx_t i = 0; i < n; i++)
            y(i, j) -= (lambda) * (lambda)*x(i, j);

    double dotProdNorm = lange(FROB_NORM, y);

    double normAcopy = lange(FROB_NORM, A_copy);

    std::cout << std::endl
              << "(||A.H*(b - bHat) - (lambda^2)*x||_F) / (||A||_F) = "
              << std::endl
              << (dotProdNorm / normAcopy) << std::endl;
}