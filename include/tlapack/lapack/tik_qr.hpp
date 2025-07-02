#ifndef TLAPACK_TIK_QR_HH
#define TLAPACK_TIK_QR_HH

#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/unmqr.hpp>

using namespace tlapack;

/// Solves tikhonov regularized least squares using QR factorization
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t>
void tik_qr(matrixA_t& A, matrixb_t& b, real_t lambda)
{
    // Initliazation for basic utilities
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;
    using vector_t = LegacyVector<T>;

    using range = pair<idx_t, idx_t>;

    Create<matrixA_t> new_matrix;
    Create<vector_t> new_vector;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    // Do a QR factorization on A
    std::vector<T> tau1_;
    auto tau1 = new_vector(tau1_, n);

    geqrf(A, tau1);

    // Apply Q1.H*b
    unmqr(LEFT_SIDE, CONJ_TRANS, A, tau1, b);

    // Initailize R augmented L
    std::vector<T> Raug_;
    auto Raug = new_matrix(Raug_, n + n, n);

    auto lam_view = slice(Raug, range{n, n + n}, range{0, n});

    lacpy(UPPER_TRIANGLE, slice(A, range{0, n}, range{0, n}), Raug);
    laset(GENERAL, real_t(0), lambda, lam_view);

    // Do a QR factorization on Raug
    std::vector<T> tau2_;
    auto tau2 = new_vector(tau2_, n);

    geqrf(Raug, tau2);

    // Initalize b augmented with zeros
    std::vector<T> baug_;
    auto baug = new_matrix(baug_, n + n, k);

    auto b_bottom = slice(baug, range{n, n + n}, range{0, k});

    lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), baug);
    laset(GENERAL, real_t(0), real_t(0), b_bottom);

    // Apply Q2.H*(Q1.H*b)
    unmqr(LEFT_SIDE, CONJ_TRANS, Raug, tau2, baug);

    // Solve R2*x = Q2.H*(Q1.H*b)
    auto bview = slice(baug, range{0, n}, range{0, k});

    trsm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1),
         slice(Raug, range{0, n}, range{0, n}), bview);

    // Copy solution to output
    lacpy(GENERAL, bview, b);
}
#endif  // TLAPACK_TIK_QR_HH