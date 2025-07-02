#include <tlapack/blas/herk.hpp>
#include <tlapack/lapack/potrf.hpp>

#include "tlapack/base/utils.hpp"

using namespace tlapack;

/// Solves tikhonov regularized least squares using cholesky method
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t,
          TLAPACK_MATRIX matrixx_t>
void tik_chol(matrixA_t& A, matrixb_t& b, real_t lambda, matrixx_t& x)
{
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;

    Create<matrixA_t> new_matrix;

    idx_t m = nrows(A);
    idx_t n = ncols(A);

    std::vector<T> H_;
    auto H = new_matrix(H_, n, n);

    // Step 1: HERK
    herk(UPPER_TRIANGLE, CONJ_TRANS, real_t(1), A, H);

    // Step 2: GEMM
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), A, b, x);

    // Step 3: A.H*A + (lambda^2)*I
    for (idx_t j = 0; j < n; j++)
        H(j, j) += lambda * lambda;

    // Step 4: POTRF
    // U -> H
    potrf(UPPER_TRIANGLE, H);

    // Step 5: TRSM
    trsm(LEFT_SIDE, UPPER_TRIANGLE, CONJ_TRANS, NON_UNIT_DIAG, real_t(1), H, x);

    // Step 6: TRSM
    trsm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1), H, x);
}