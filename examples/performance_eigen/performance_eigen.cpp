/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <chrono>  // for high_resolution_clock
#include <iostream>

// From Eigen
#include <Eigen/Dense>
#include <Eigen/Householder>

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
    using idx_t = Eigen::Index;
    using ref_T = long double;

    // Number of eigenvalues to be used to compute error
    const idx_t N = n;
    const idx_t Nshow = 2;

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

    std::cout << "--------------" << std::endl;
    std::cout << "1. Eigen -----" << std::endl;
#if defined(EIGEN_USE_MKL_ALL)
    std::cout << "Using MKL" << std::endl;
#elif defined(EIGEN_USE_BLAS)
    std::cout << "Using MKL BLAS" << std::endl;
#endif

    // Compute eigenvalues using Eigen
    {
        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        Eigen::Matrix<std::complex<T>, -1, 1> lambda(n);
        Eigen::Matrix<T, -1, -1> matrixT(n, n);
        Eigen::Matrix<T, -1, -1> U(n, n);

        if (compute_backwardError) {
            Eigen::RealSchur<decltype(A)> solverSchur(A, true);
            matrixT = solverSchur.matrixT();
            U = solverSchur.matrixU();

            idx_t i = 0;
            while (i < n) {
                if (i == n - 1 || matrixT(i + 1, i) == T(0)) {
                    lambda[i] = matrixT(i, i);
                    ++i;
                }
                else {
                    T p = T(0.5) * (matrixT(i, i) - matrixT(i + 1, i + 1));
                    T z;

                    // Compute z = sqrt(abs(p * p + matrixT(i+1, i) * matrixT(i,
                    // i+1))); without overflow
                    {
                        T t0 = matrixT(i + 1, i);
                        T t1 = matrixT(i, i + 1);
                        T maxval = std::max(abs(p), std::max(abs(t0), abs(t1)));
                        t0 /= maxval;
                        t1 /= maxval;
                        T p0 = p / maxval;
                        z = maxval * sqrt(abs(p0 * p0 + t0 * t1));
                    }

                    lambda[i] = std::complex<T>(matrixT(i + 1, i + 1) + p, z);
                    lambda[i + 1] =
                        std::complex<T>(matrixT(i + 1, i + 1) + p, -z);

                    i += 2;
                }
            }
        }
        else {
            Eigen::EigenSolver<decltype(A)> solver(A, false);
            lambda = solver.eigenvalues();
        }

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // // Compute backward error before reordering eigenvalues
        // Eigen::Matrix<T,-1,1> bw_error(2);
        // if( compute_backwardError )
        // {
        //     auto V = solverSchur.matrixU() * solver.eigenvectors();
        //     auto R0 = V * lambda.asDiagonal() - A * V;
        //     auto R1 = V * lambda.asDiagonal() * V.inverse() - A;

        //     bw_error[0] = R0.norm() / ( A.norm() * V.norm() );
        //     bw_error[1] = R1.norm() / A.norm();
        // }

        // Reorder eigenvalues
        std::sort(lambda.data(), lambda.data() + lambda.size(),
                  complex_comparator<T>);

        // Compute error
        Eigen::Matrix<std::complex<T>, -1, 1> error(N);
        for (idx_t i = 0; i < N; ++i)
            error[i] = lambda[i] - exact_eig[i];

        // Output
        std::cout << "First eigenvalues:" << std::endl
                  << lambda.segment(0, Nshow) << std::endl
                  << "time = " << elapsed.count() * 1.0e-9 << " s" << std::endl;
        if (compute_forwardError)
            std::cout << "||fl(lambda)-lambda||/||lambda|| = "
                      << error.norm() / exact_eig.norm() << std::endl;
        if (compute_backwardError) {
            // Check the Schur factorization is correct
            Eigen::Matrix<T, -1, -1> R0 =
                U.adjoint() * U - Eigen::Matrix<T, -1, -1>::Identity(n, n);
            Eigen::Matrix<T, -1, -1> R1 =
                U * U.adjoint() - Eigen::Matrix<T, -1, -1>::Identity(n, n);
            Eigen::Matrix<T, -1, -1> R2 = U * matrixT * U.adjoint() - A;

            std::cout << "||U^H U - I||/||I|| = " << R0.norm() / sqrt(n * T(1))
                      << std::endl
                      << "||U U^H - I||/||I|| = " << R1.norm() / sqrt(n * T(1))
                      << std::endl
                      << "||U T U^H - A||/||A|| = " << R2.norm() / A.norm()
                      << std::endl;
            // << "||V Lambda - A V||/(||A|| ||V||) = " << bw_error[0] <<
            // std::endl
            // << "||V Lambda V^{-1} - A||/||A|| = " << bw_error[1] <<
            // std::endl;
        }
    }
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
