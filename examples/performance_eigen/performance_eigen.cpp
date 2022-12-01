/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <iostream>
#include <chrono>   // for high_resolution_clock

// From Eigen
#include <Eigen/Dense>
#include <Eigen/Householder>

template <class T>
bool complex_comparator(const std::complex<T> &a, const std::complex<T> &b) {
    return std::real(a) == std::real(b) ? std::imag(a) > std::imag(b) : std::real(a) > std::real(b);
}

//------------------------------------------------------------------------------
template <typename T>
void run( Eigen::Index n, bool run_reference )
{
    using idx_t = Eigen::Index;
    using ref_T = long double;

    // Number of eigenvalues to be used to compute error
    const idx_t N = n;
    const idx_t Nshow = 2;

    // Matrices
    Eigen::Matrix<ref_T,-1,-1> A_ref(n,n);
    Eigen::Matrix<T,-1,-1> A(n,n);

    // Generate a random matrix in A
    srand(7435);
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i) {
            A(i,j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
            A_ref(i,j) = A(i,j);
        }

    // Reference eigenvalues
    Eigen::Matrix<std::complex<ref_T>,-1,1> ref_eig(n);
    if( run_reference )
    {
        Eigen::EigenSolver<decltype(A_ref)> solver( A_ref );
        solver.compute( A_ref, false );
        ref_eig = solver.eigenvalues();

        // Reorder eigenvalues
        std::sort(ref_eig.data(), ref_eig.data() + ref_eig.size(), complex_comparator< ref_T >);

        // Output
        std::cout   << "Reference:" << std::endl
                    << "First eigenvalues:" << std::endl
                    << ref_eig.segment(0,Nshow) << std::endl;
    }

    std::cout << "-x-x-x-x-x-x-x-x-" << std::endl;
    #ifdef EIGEN_USE_MKL_ALL
        std::cout << "-x-x-x-MKL-x-x-x-" << std::endl;
    #endif

    // Compute eigenvalues using Eigen
    {
        // Record start time
        auto start = std::chrono::high_resolution_clock::now();
        
            Eigen::Matrix<std::complex<T>,-1,1> eigen_eig(n);
            Eigen::EigenSolver<decltype(A)> solver;
            solver.compute( A, false );
            eigen_eig = solver.eigenvalues();
        
        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Reorder eigenvalues
        std::sort(eigen_eig.data(), eigen_eig.data() + eigen_eig.size(), complex_comparator< T >);

        // Compute error
        Eigen::Matrix<std::complex<ref_T>,-1,1> error(N);
        for( idx_t i = 0; i < N; ++i )
            error[i] = (std::complex<ref_T>) eigen_eig[i] - ref_eig[i];

        // Output
        std::cout   << "Using Eigen:" << std::endl
                    << "First eigenvalues:" << std::endl
                    << eigen_eig.segment(0,Nshow) << std::endl
                    << "Error in the eigenvalues: " << error.norm() / ref_eig.norm() << std::endl
                    << "time = " << elapsed.count() * 1.0e-9 << " s" << std::endl;
    }
}

int main( int argc, char** argv )
{
    // Default arguments
    const Eigen::Index n = ( argc < 2 ) ? 2 : atoi( argv[1] );
    const bool run_reference = ( argc < 3 ) ? true : (atoi( argv[2] ) != 0);

    run<float>( n, run_reference );
    run<double>( n, run_reference );

    return 0;
}
