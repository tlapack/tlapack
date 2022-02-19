/// @file example_gemm.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <vector>
#include <iostream>
#include <chrono>   // for high_resolution_clock

// <T>LAPACK
#include <plugins/tlapack_legacyArray.hpp>
#include <tlapack.hpp>

// LAPACKE
extern "C" {
    #include <lapacke.h>
}
template<typename T>
inline constexpr
lapack_int LAPACKE_xpotrf( int matrix_layout, char uplo, lapack_int n, T* a, lapack_int lda ) {
    return 1;
}
template<>
inline
lapack_int LAPACKE_xpotrf<float>( int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda ) {
    return LAPACKE_spotrf( matrix_layout, uplo, n, a, lda );
}
template<>
inline
lapack_int LAPACKE_xpotrf<double>( int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda ) {
    return LAPACKE_dpotrf( matrix_layout, uplo, n, a, lda );
}
template<>
inline
lapack_int LAPACKE_xpotrf<std::complex<float>>( int matrix_layout, char uplo, lapack_int n, std::complex<float>* a, lapack_int lda ) {
    return LAPACKE_cpotrf( matrix_layout, uplo, n, reinterpret_cast<float _Complex*>(a), lda );
}
template<>
inline
lapack_int LAPACKE_xpotrf<std::complex<double>>( int matrix_layout, char uplo, lapack_int n, std::complex<double>* a, lapack_int lda ) {
    return LAPACKE_zpotrf( matrix_layout, uplo, n, reinterpret_cast<double _Complex*>(a), lda );
}

using idx_t = BLAS_SIZE_T;

//------------------------------------------------------------------------------
template <typename T>
void run( idx_t n )
{
    using namespace lapack;
    using real_t = real_type<T>;

    // Matrix A
    std::vector<T> _A( n*n );
    legacyMatrix<T> A( n, n, &_A[0], n );
    
    // Fill A with random entries
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i)
            A(i,j) = static_cast<float>( rand() )
                    / static_cast<float>( RAND_MAX );
    }
    
    // Turn the upper part of A into a symmetric positive definite matrix
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < j; ++i)
            A(i,j) = T(0.5) * ( A(i,j) + A(j,i) );
        A(j,j) += n;
    }

    // 1) Using <T>LAPACK interface:
    {
        std::vector<T> _U( n*n );
        legacyMatrix<T> U( n, n, &_U[0], n );

        // Put garbage on _U
        for (idx_t j = 0; j < n*n; ++j)
            _U[j] = blas::make_scalar<real_t>( static_cast<float>( 0xDEADBEEF ), static_cast<float>( 0xDEADBEEF ) );

        // _U receives the upper part of A
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i <= j; ++i)
                _U[i+j*n] = A(i,j);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

            struct { idx_t nb = 64; } opts;
            int info = potrf( upper_triangle, U, opts );
        
        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
        if( info != 0 ) {
            std::cout << "Cholesky ended with info " << info << std::endl;
        }

        // Solve U^H U R = A
        std::vector<T> _R( n*n );
        legacyMatrix<T> R( n, n, &_R[0], n );
        lacpy( general_matrix, A, R );
        potrs( upper_triangle, U, R );

        // error = ||R-Id||_F / ||Id||_F
        for (idx_t i = 0; i < n; ++i)
            R(i,i) -= T(1);
        real_t error = lange( frob_norm, R ) / std::sqrt(n);

        // Output
        std::cout << "Using <T>LAPACK:" << std::endl
                << "U^H U R = A   =>   ||R-Id||_F / ||Id||_F = " << error << std::endl
                << "time = " << elapsed.count() * 1.0e-6 << " ms" << std::endl;
    }

    // 2) Using LAPACKE:
    {
        std::vector<T> U( n*n );

        // Put garbage on U
        for (idx_t j = 0; j < n*n; ++j)
            U[j] = blas::make_scalar<real_t>( static_cast<float>( 0xDEADBEEF ), static_cast<float>( 0xDEADBEEF ) );

        // U receives the upper part of A
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i <= j; ++i)
                U[i+j*n] = A(i,j);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

            lapack_int info = LAPACKE_xpotrf<T>( LAPACK_COL_MAJOR, 'U', n, &U[0], n );
        
        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
        if( info != 0 ) {
            std::cout << "Cholesky ended with info " << info << std::endl;
        }

        // Solve U^H U R = A
        std::vector<T> _R( n*n );
        legacyMatrix<T> R( n, n, &_R[0], n );
        lacpy( general_matrix, A, R );
        potrs( upper_triangle, legacyMatrix<T>( n, n, &U[0], n ), R );

        // error = ||R-Id||_F / ||Id||_F
        for (idx_t i = 0; i < n; ++i)
            R(i,i) -= T(1);
        real_t error = lange( frob_norm, R ) / std::sqrt(n);

        // Output
        std::cout << "Using LAPACKE:" << std::endl
                << "U^H U R = A   =>   ||R-Id||_F / ||Id||_F = " << error << std::endl
                << "time = " << elapsed.count() * 1.0e-6 << " ms" << std::endl;
    }
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    idx_t n;

    // Default arguments
    n = ( argc < 2 ) ? 100 : atoi( argv[1] );

    srand( 3 ); // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;
    
    std::cout << "run< float >( " << n << " )" << std::endl;
    run< float >( n );
    std::cout << "-----------------------" << std::endl;
    
    std::cout << "run< double >( " << n << " )" << std::endl;
    run< double >( n );
    std::cout << "-----------------------" << std::endl;
    
    std::cout << "run< complex<float> >( " << n << " )" << std::endl;
    run< std::complex<float> >( n );
    std::cout << "-----------------------" << std::endl;
    
    std::cout << "run< complex<double> >( " << n << " )" << std::endl;
    run< std::complex<double> >( n );
    std::cout << "-----------------------" << std::endl;

    return 0;
}
