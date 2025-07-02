#ifndef TLAPACK_TIK_SVD
#define TLAPACK_TIK_SVD

#include <tlapack/lapack/bidiag.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/svd_qr.hpp>
#include <tlapack/lapack/ungbr.hpp>
#include <tlapack/lapack/unmqr.hpp>

using namespace tlapack;

template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t>
void tik_svd(matrixA_t& A, matrixb_t& b, real_t lambda)
{
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;

    Create<matrixA_t> new_matrix;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    using range = pair<idx_t, idx_t>;

    std::vector<T> tauv(n);
    std::vector<T> tauw(n);

    // Bidiagonal decomposition
    bidiag(A, tauv, tauw);

    // Apply Q1ᵀ to b using unmqr
    //
    // Note: it is possible to use b for tmp1 and output x in b,
    // this would remove arrays btmp1 and x. Right now, we chose to have
    // the same interface for all tikhonov functions
    std::vector<T> btmp1_;
    auto btmp1 = new_matrix(btmp1_, m, k);
    lacpy(GENERAL, b, btmp1);
    unmqr(LEFT_SIDE, CONJ_TRANS, A, tauv, btmp1);

    // Slice top n rows: Q1ᵀ b
    // std::vector<T> x_;
    // auto x = new_matrix(x_, n, k);
    auto x = slice(b, range{0, n}, range{0, k});

    lacpy(GENERAL, slice(btmp1, range{0, n}, range{0, k}), x);

    // Extract diagonal and superdiagonal
    std::vector<real_t> d(n);
    std::vector<real_t> e(n - 1);
    for (idx_t j = 0; j < n; ++j)
        d[j] = real(A(j, j));
    for (idx_t j = 0; j < n - 1; ++j)
        e[j] = real(A(j, j + 1));

    // Construct P1
    // Note: it is possible to store P1 in A
    std::vector<T> P1_;
    auto P1 = new_matrix(P1_, n, n);
    lacpy(UPPER_TRIANGLE, slice(A, range{0, n}, range{0, n}), P1);
    ungbr_p(n, P1, tauw);

    // Allocate and initialize Q2 and P2t
    std::vector<T> Q2_;
    auto Q2 = new_matrix(Q2_, n, n);
    std::vector<T> P2t_;
    auto P2t = new_matrix(P2t_, n, n);
    const real_t zero(0);
    const real_t one(1);
    laset(Uplo::General, zero, one, Q2);
    laset(Uplo::General, zero, one, P2t);

    // Note that it would be better to input "x" which is "Q1ᵀb" instead of
    // computing "Q2" and then applying "Q2ᵀ". This would be done as follows:
    //
    // int err = svd_qr(Uplo::Upper, true, true, d, e, x, P2t);
    //
    // this does not work though.

    int err = svd_qr(Uplo::Upper, true, true, d, e, Q2, P2t);

    // Apply Q2ᵀ
    std::vector<T> x2_;
    auto x2 = new_matrix(x2_, n, k);
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), Q2, x, real_t(0), x2);

    // lacpy(GENERAL, x, x2);

    // printMatrix(x2);

    // This is what was changed from least_squares_svd in order to do tik_svd
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < k; ++i)
            // x2(j, i) /= d[j];
            x2(j, i) *= (d[j] / ((d[j] * d[j]) + (lambda * lambda)));

    // Apply P2tᵀ
    std::vector<T> x3_;
    auto x3 = new_matrix(x3_, n, k);
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), P2t, x2, real_t(0), x3);

    // Apply P1ᵀ
    std::vector<T> x4_;
    auto x4 = new_matrix(x4_, n, k);
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), P1, x3, real_t(0), x4);

    // Final result

    lacpy(GENERAL, x4, x);
}

#endif  // TLAPACK_TIK_SVD