/// @file example_geqp2.cpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
//
// Copyright (c) 2026, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// GEQP2 computes a QR factorization of a real M-by-N
// matrix A, using Drmac's algorithm for.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/swap.hpp>
#include <tlapack/lapack/geqp2.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/larfb.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/lapack/unmqr.hpp>

// C++ Headers
#include <chrono>  // for high_resolution_clock
#include <random>

enum class PermuteTarget { Rows, Columns };

/**
 * @brief Permutes either the rows or columns of a matrix in-place using a
 * SPECIFIC permutation vector.
 *
 * @note The `perm` vector is taken by value because the cycle-following
 * algorithm destructively modifies the tracking array to achieve O(1) auxiliary
 * space.
 */
template <typename matrix_t, typename idx_vector_t>
void permute_matrix(matrix_t& A,
                    PermuteTarget target,
                    idx_vector_t& target_order)
{
    using idx_t = tlapack::size_type<matrix_t>;

    // constants
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    bool is_row = (target == PermuteTarget::Rows);
    idx_t primary_dim = is_row ? m : n;

    std::vector<bool> visited(primary_dim, false);
    for (idx_t i = 0; i < primary_dim; ++i) {
        // Skip if already processed or already in the correct spot
        if (visited[i] || target_order[i] == i) {
            continue;
        }

        idx_t current = i;
        while (!visited[current]) {
            visited[current] = true;
            idx_t next_node = target_order[current];

            if (!visited[next_node]) {
                // Swap entire row or column vectors
                if (is_row) {
                    auto current_row = tlapack::row(A, current);
                    auto next_row = tlapack::row(A, next_node);

                    tlapack::swap(current_row, next_row);
                }
                else {
                    auto current_col = tlapack::col(A, current);
                    auto next_col = tlapack::col(A, next_node);

                    tlapack::swap(current_col, next_col);
                }

                current = next_node;
            }
        }
    }
}

//-----------------------------------------------------------------------
// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;

    // constants
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
    std::cout << std::endl;
}

//---------------------------------------------------------------------------
template <typename matrixA_t, typename matrixV_t, typename vectorT_t>
inline void check(matrixA_t& A, matrixV_t& V, const vectorT_t& tau)

{
    using T = typename tlapack::type_t<matrixA_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrixA_t>;

    // constants
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    // Functors for creating new matrices
    tlapack::Create<matrixA_t> new_matrix;

    // Compute the Frobenius Norm of A
    real_t normA = tlapack::lange(tlapack::FROB_NORM, A);

    // Copy the upper-triangular part of V into R.
    // Zero the full matrix first so lower triangle does not contain
    // garbage.
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    tlapack::laset(tlapack::GENERAL, T(0.0), T(0.0), R);
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, V, R);

    // Form the explicit orthogonal matrix Q from the Householder factors.
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, n);
    tlapack::lacpy(tlapack::GENERAL, V, Q);
    tlapack::ung2r(Q, tau);

    // Compute the reconstruction accuracy ||A - Q*R||_F.
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(-1.0), Q, R,
                  T(1.0), A);

    real_t recon_error = tlapack::lange(tlapack::FROB_NORM, A);

    // Compute the orthogonality error ||I - Q^H Q||_F.
    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);

    tlapack::laset(tlapack::UPPER_TRIANGLE, T(0.0), T(1.0), work);

    tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans, T(-1.0), Q, Q,
                  T(1.0), work);
    real_t orthogonality_error = tlapack::lange(tlapack::FROB_NORM, work);
    // Prints the reconstruction error and orthogonality error
    std::cout << std::endl
              << "||A - Q*R||_F / ||A||_F = " << recon_error / normA
              << std::endl;
    std::cout << "||I - Q^H Q||_F = "
              << orthogonality_error / static_cast<real_t>(n) << std::endl;
}

//----------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t r, size_t k)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Initialize the random engine ONCE (e.g., at application startup)
    std::random_device rd;
    std::mt19937 gen(rd());

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Arrays
    std::vector<idx_t> perm(n - k);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<idx_t> col_order(n);
    std::iota(col_order.begin(), col_order.end(), 0);
    std::vector<idx_t> row_order(m);
    std::iota(row_order.begin(), row_order.end(), 0);
    std::vector<T> tau(n);
    std::vector<real_t> vn1(n);
    std::vector<real_t> vn2(n);

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_orig_;
    auto A_orig = new_matrix(A_orig_, m, n);
    std::vector<T> S_;
    auto S = new_matrix(S_, m, r);
    std::vector<T> C_;
    auto C = new_matrix(C_, r, n);

    // Slices
    // Define A_1 := A[0:m,0:k] which is the first k columns of A
    auto A_1 = slice(A, range(0, m), range(0, k));
    // Define A_2 := A[0:m,k:n]
    auto A_2 = slice(A, range(0, m), range(k, n));
    // Define A_22 := A[k:m,k:n]
    auto A_22 = slice(A, range(k, m), range(k, n));
    // Define R_12 := A[0:k,k:n]
    auto R_12 = slice(A, range(0, k), range(k, n));
    // Define tau_head := tau[0:k]
    auto tau_head = tlapack::slice(tau, std::pair{0, k});
    // Define tau_tail := tau[k:n]
    auto tau_tail = tlapack::slice(tau, std::pair{k, n});

    // constant
    const real_t eps = tlapack::ulp<real_t>();
    // Initialize the Rank
    idx_t rank = 0;

    // Initialize arrays with junk
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A(i, j) = static_cast<T>(0xDEADBEEF);
        }
        tau[j] = static_cast<T>(0xFFBADD11);
    }

    // Create random matrices in submatrices of S and C
    // Generate a random matrix in S
    for (idx_t j = 0; j < r; ++j)
        for (idx_t i = 0; i < m; ++i)
            if constexpr (tlapack::is_complex<T>)
                S(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            else
                S(i, j) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));

    // Generate a random matrix in C
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < r; ++i)
            if constexpr (tlapack::is_complex<T>)
                C(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            else
                C(i, j) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));

    // 1) Compute A = S * C

    // Perform Matrix multiplication A = 1.0*S*C + 0.0*A
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(1.0), S, C,
                  T(0.0), A);

    /*
    // Fill A with random values
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            if constexpr (tlapack::is_complex<T>)
                A(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            else
                A(i, j) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));
    // Perturb the first n-r+1 columns based on the first column so that the
    first n-r+1
    // columns are linearly dependent
    for (idx_t j = 0; j < n - r + 1; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            T x = 0.0;
            if constexpr (tlapack::is_complex<T>)
                x = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            else
                x = T(static_cast<float>(rand()) /
                      static_cast<float>(RAND_MAX));
            A(i, j) = A(i, 1) + x * eps;
        }
    }
    */

    // 2) Apply Two-Sided Random Permutations to Matrix A

    // Creates randomized order for the vectors col_order and row_order
    std::shuffle(col_order.begin(), col_order.end(), gen);
    std::shuffle(row_order.begin(), row_order.end(), gen);

    // Permutes the columns of matrix A
    permute_matrix(A, PermuteTarget::Columns, col_order);
    // Permutes the rows of matrix A
    permute_matrix(A, PermuteTarget::Rows, row_order);

    // Copy A to A_orig
    tlapack::lacpy(tlapack::GENERAL, A, A_orig);

    // 3) Compute QR Factorization of A_1
    // Compute the QR Factorization of the first k columns
    tlapack::geqr2(A_1, tau_head);

    // 4) Apply Q_1^H to the Trailing Block A_2
    tlapack::unmqr(tlapack::Side::Left, tlapack::Op::ConjTrans, A_1, tau_head,
                   A_2);

    // 5) Compute Rank-Revealing QR (Drmac) on A_22
    geqp2(A_22, tau_tail, perm, vn1, vn2);

    // 6) Permutes Upper Right Block R_12 per the rank revealing algorithm of
    // Drmac
    permute_matrix(R_12, PermuteTarget::Columns, perm);

    // 7) Evaluate Numerical Rank
    // Set absolute threshold: tolerance = max(m,n) * epsilon * |R(0, 0)|
    real_t tol = std::max(m, n) * eps * std::abs(A(0, 0));
    for (idx_t i = 0; i < std::min(m, n); i++) {
        if (std::abs(A(i, i)) > tol) {
            rank++;
        }
        else {
            break;
        }
    }
    // Output numerical rank results
    std::cout << std::endl << "rank of A = " << rank;

    // 8) Compute Backward Error Validation

    //  Define A_orig_2 := A_orig[0:m,k:n]
    auto A_orig_2 = slice(A_orig, range(0, m), range(k, n));
    // Permute the original matrix to produce AP before sendind it to the check
    permute_matrix(A_orig_2, PermuteTarget::Columns, perm);

    check(A_orig, A, tau);
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n, r, k;

    // Default arguments
    m = (argc < 2) ? 7 : atoi(argv[1]);
    n = (argc < 3) ? 7 : atoi(argv[2]);
    r = (argc < 4) ? 5 : atoi(argv[3]);
    k = (argc < 5) ? 3 : atoi(argv[4]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d, %d, %d )", m, n, r, k);
    run<float>(m, n, r, k);
    std::cout << std::endl;
    printf("-----------------------\n");

    printf("run< double >( %d, %d, %d, %d )", m, n, r, k);
    run<double>(m, n, r, k);
    std::cout << std::endl;
    printf("-----------------------\n");

    printf("run< long double >( %d, %d, %d, %d )", m, n, r, k);
    run<long double>(m, n, r, k);
    std::cout << std::endl;
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d, %d, %d )", m, n, r, k);
    run<std::complex<float> >(m, n, r, k);
    std::cout << std::endl;
    std::cout << "-----------------------" << std::endl;

    printf("run< complex<double> >( %d, %d, %d, %d )", m, n, r, k);
    run<std::complex<double> >(m, n, r, k);
    std::cout << std::endl;
    std::cout << "-----------------------" << std::endl;

    printf("run< complex<long double> >( %d, %d, %d, %d )", m, n, r, k);
    run<std::complex<long double> >(m, n, r, k);
    std::cout << std::endl;
    std::cout << "-----------------------" << std::endl;
    return 0;
}