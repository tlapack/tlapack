/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/eigen.hpp>
#ifdef USE_MDSPAN_DATA
    #include <tlapack/plugins/mdspan.hpp>
#endif

// <T>LAPACK
#include <tlapack/blas/nrm2.hpp>
#include <tlapack/lapack/gehrd.hpp>
#include <tlapack/lapack/multishift_qr.hpp>
#include <tlapack/lapack/unghr.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Householder>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>

template <class T>
bool complex_comparator(const std::complex<T>& a, const std::complex<T>& b)
{
    return std::real(a) == std::real(b) ? std::imag(a) > std::imag(b)
                                        : std::real(a) > std::real(b);
}

//------------------------------------------------------------------------------
template <typename T>
void run(int n,
         bool compute_forwardError,
         bool compute_backwardError,
         int matrix_type)
{
    using namespace tlapack;
    using ref_T = long double;

    // Number of eigenvalues to be used to compute error
    const int N = n;
    const int Nshow = 2;

    // Matrices
    Eigen::Matrix<T, -1, -1> A(n, n);
    Eigen::Matrix<std::complex<T>, -1, 1> exact_eig(n);

    // Generate a random matrix in A
    if (matrix_type == 0) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                A(i, j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }
    else {
        // Generate a random ill-conditioned matrix
        {
            // D1 is a diagonal matrix containing the desired eigenvalues
            Eigen::Matrix<T, -1, 1> d1(n);
            for (int i = 0; i < n; ++i) {
                d1[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
                exact_eig[i] = d1[i];
            }

            // D2 is a diagonal matrix, the condition of this matrix will become
            // the condition of the eigenvector matrix
            Eigen::Matrix<T, -1, 1> d2(n);
            for (int i = 0; i < n; ++i)
                d2[i] = std::pow(2, i % matrix_type);

            // Q1 and Q2 are random unitary matrices
            Eigen::Matrix<T, -1, -1> Q1(n, n);
            Eigen::Matrix<T, -1, -1> Q2(n, n);
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    Q1(i, j) =
                        static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
                    Q2(i, j) =
                        static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
                }
            }
            Q1 = Eigen::HouseholderQR<decltype(A)>(Q1).householderQ();
            Q2 = Eigen::HouseholderQR<decltype(A)>(Q2).householderQ();

            // A = Q2^* D2^-1 Q1^* D1 Q1 D2 Q2
            A = Q2.adjoint() * d2.asDiagonal().inverse() * Q1.adjoint() *
                d1.asDiagonal() * Q1 * d2.asDiagonal() * Q2;
        }
    }

    // Reference eigenvalues
    if (compute_forwardError) {
        if (matrix_type == 0) {
            std::cout << "0. Eigen (reference solution using long double)"
                      << std::endl;

            Eigen::Matrix<ref_T, -1, -1> A_ref(n, n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A_ref(i, j) = A(i, j);

            Eigen::EigenSolver<decltype(A_ref)> solver(A_ref);
            solver.compute(A_ref, false);
            Eigen::Matrix<std::complex<ref_T>, -1, 1> ref_eig =
                solver.eigenvalues();
            for (int i = 0; i < n; ++i)
                exact_eig[i] = ref_eig[i];
        }

        // Reorder eigenvalues
        std::sort(exact_eig.data(), exact_eig.data() + exact_eig.size(),
                  complex_comparator<T>);

        // Output
        std::cout << "First eigenvalues:" << std::endl
                  << exact_eig.segment(0, Nshow) << std::endl;
    }

    std::cout << "-------------------" << std::endl;
    std::cout << "1. <T>LAPACK ------" << std::endl;
    std::cout << "Using Eigen::Matrix" << std::endl;

    // Compute eigenvalues using <T>LAPACK with Eigen matrices
    {
        using idx_t = Eigen::Index;

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        // Hessenberg reduction
        Eigen::Matrix<T, -1, -1> H = A;
        Eigen::Matrix<T, -1, 1> tau(n);
        gehrd(0, n, H, tau);

        // Save reflectors elsewhere (in Q) to make a true Hessenberg matrix
        Eigen::Matrix<T, -1, -1> Q(n, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i) {
                Q(i, j) = H(i, j);
                H(i, j) = T(0);
            }

        // Compute eigenvalues
        Eigen::Matrix<std::complex<T>, -1, 1> tlapack_eig(n);
        Eigen::Matrix<T, -1, -1> U(n, n);
        Eigen::Matrix<T, -1, -1> matrixT = H;

        // Initialize U
        if (compute_backwardError) {
            unghr(0, n, Q, tau);
            U = Q;
        }
        else
            U.setIdentity();

        int n_aed, n_sweep, n_shifts_total;
        {
            FrancisOpts<idx_t> opts;

            multishift_qr(compute_backwardError, compute_backwardError, 0, n,
                          matrixT, tlapack_eig, U, opts);

            n_aed = opts.n_aed;
            n_sweep = opts.n_sweep;
            n_shifts_total = opts.n_shifts_total;
        }

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Clean the lower triangular part that was used a workspace
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                matrixT(i, j) = T(0);

        // Reorder eigenvalues
        std::sort(tlapack_eig.data(), tlapack_eig.data() + tlapack_eig.size(),
                  complex_comparator<T>);

        // Compute error
        Eigen::Matrix<std::complex<T>, -1, 1> error(N);
        for (idx_t i = 0; i < N; ++i)
            error[i] = tlapack_eig[i] - exact_eig[i];

        // Output
        std::cout << "First eigenvalues:" << std::endl
                  << tlapack_eig.segment(0, Nshow) << std::endl
                  << "time = " << elapsed.count() * 1.0e-9 << " s" << std::endl
                  << "n_aed = " << n_aed << std::endl
                  << "n_sweep = " << n_sweep << std::endl
                  << "n_shifts_total = " << n_shifts_total << std::endl;
        if (compute_forwardError)
            std::cout << "||fl(lambda)-lambda||/||lambda|| = "
                      << error.norm() / exact_eig.norm() << std::endl;
        if (compute_backwardError) {
            // Check the Schur factorization is correct
            Eigen::Matrix<T, -1, -1> R0 =
                U.adjoint() * U - Eigen::Matrix<T, -1, -1>::Identity(n, n);
            Eigen::Matrix<T, -1, -1> R1 =
                U * U.adjoint() - Eigen::Matrix<T, -1, -1>::Identity(n, n);
            Eigen::Matrix<T, -1, -1> R2 =
                Q.adjoint() * Q - Eigen::Matrix<T, -1, -1>::Identity(n, n);
            Eigen::Matrix<T, -1, -1> R3 =
                Q * Q.adjoint() - Eigen::Matrix<T, -1, -1>::Identity(n, n);
            Eigen::Matrix<T, -1, -1> R4 = U * matrixT * U.adjoint() - A;

            std::cout << "||U^H U - I||/||I|| = " << R0.norm() / sqrt(n * T(1))
                      << std::endl
                      << "||U U^H - I||/||I|| = " << R1.norm() / sqrt(n * T(1))
                      << std::endl
                      << "||U T U^H - A||/||A|| = " << R4.norm() / A.norm()
                      << std::endl;
            // << "||Q^H Q - I||/||I|| = " << R2.norm() / sqrt(n*T(1)) <<
            // std::endl
            // << "||Q Q^H - I||/||I|| = " << R3.norm() / sqrt(n*T(1)) <<
            // std::endl;
        }
    }

#ifdef USE_MDSPAN_DATA

    std::cout << "--------------------" << std::endl;
    std::cout << "2. <T>LAPACK -------" << std::endl;
    std::cout << "Using kokkos::mdspan" << std::endl;

    // Compute eigenvalues using <T>LAPACK with kokkos::mdspan matrices
    {
        using idx_t = size_t;
        using mdspan_matrix =
            std::experimental::mdspan<T, std::experimental::dextents<idx_t, 2>,
                                      std::experimental::layout_left>;
        using mdspan_vector =
            std::experimental::mdspan<T,
                                      std::experimental::dextents<idx_t, 1> >;
        using mdspan_complexvector =
            std::experimental::mdspan<std::complex<T>,
                                      std::experimental::dextents<idx_t, 1> >;

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        // (1) Hessenberg reduction

        std::vector<T> H_(A.data(), A.data() + n * n);
        mdspan_matrix H(H_.data(), n, n);

        std::vector<T> tau_(n);
        mdspan_vector tau(tau_.data(), n);

        tlapack::gehrd(0, n, H, tau);

        // Save reflectors elsewhere (in Q) to make a true Hessenberg matrix
        std::vector<T> Q_(H.data(), H.data() + n * n);
        mdspan_matrix Q(Q_.data(), n, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                H(i, j) = T(0);

        // (2) Compute eigenvalues

        std::vector<std::complex<T> > tlapack_eig_(n);
        mdspan_complexvector tlapack_eig(tlapack_eig_.data(), n);

        std::vector<T> U_(n * n);
        mdspan_matrix U(U_.data(), n, n);

        std::vector<T> matrixT_(&H_[0], &H_[0] + n * n);
        mdspan_matrix matrixT(matrixT_.data(), n, n);

        // Initialize U
        if (compute_backwardError) {
            unghr(0, n, Q, tau);
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < n; ++i)
                    U(i, j) = Q(i, j);
        }
        else {
            for (idx_t j = 0; j < n; ++j)
                U(j, j) = T(1);
        }

        int n_aed, n_sweep, n_shifts_total;
        {
            FrancisOpts<idx_t> opts;

            multishift_qr(compute_backwardError, compute_backwardError, 0, n,
                          matrixT, tlapack_eig, U, opts);

            n_aed = opts.n_aed;
            n_sweep = opts.n_sweep;
            n_shifts_total = opts.n_shifts_total;
        }

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Clean the lower triangular part that was used a workspace
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                matrixT(i, j) = T(0);

        // Reorder eigenvalues
        std::sort(&tlapack_eig_[0], &tlapack_eig_[0] + tlapack_eig.size(),
                  complex_comparator<T>);

        // Compute error
        std::vector<std::complex<T> > error_(N);
        mdspan_complexvector error(error_.data(), N);
        for (idx_t i = 0; i < N; ++i)
            error[i] = tlapack_eig[i] - exact_eig[i];

        // Output
        std::cout << "First eigenvalues:" << std::endl;
        for (idx_t i = 0; i < Nshow; ++i)
            std::cout << tlapack_eig[i] << std::endl;
        std::cout << "time = " << elapsed.count() * 1.0e-9 << " s" << std::endl
                  << "n_aed = " << n_aed << std::endl
                  << "n_sweep = " << n_sweep << std::endl
                  << "n_shifts_total = " << n_shifts_total << std::endl;
        if (compute_forwardError)
            std::cout << "||fl(lambda)-lambda||/||lambda|| = "
                      << nrm2(error) / nrm2(exact_eig) << std::endl;
    }

#endif
}

int main(int argc, char** argv)
{
    // Default arguments
    const int n = (argc < 2) ? 2 : atoi(argv[1]);
    const bool compute_forwardError = (argc < 3) ? true : (atoi(argv[2]) != 0);
    const bool compute_backwardError = (argc < 4) ? true : (atoi(argv[3]) != 0);
    const int matrix_type = (argc < 5) ? 0 : atoi(argv[4]);
    const int seed = (argc < 6) ? 3 : atoi(argv[5]);

    srand(seed);

    std::cout << "# Single precision:" << std::endl;
    run<float>(n, compute_forwardError, compute_backwardError, matrix_type);
    std::cout << std::endl;

    std::cout << "# Double precision:" << std::endl;
    run<double>(n, compute_forwardError, compute_backwardError, matrix_type);
    std::cout << std::endl;

    return 0;
}
