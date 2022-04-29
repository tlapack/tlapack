/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <iostream>

// Must be loaded in the following order
#include <plugins/tlapack_eigen.hpp>
#include <tlapack.hpp>

#include <Eigen/Dense>
#include <Eigen/Householder>

int main( int argc, char** argv )
{
    using std::size_t;
    using pair = std::pair<size_t,size_t>;
    using namespace tlapack;
    using Eigen::Matrix;

    // Constants
    const size_t m = 5;
    const size_t n = 3;

    // Input data
    Matrix<float, m, n> A;
    A << 1,  2,  3,
         4,  5,  6,
         7,  8,  9,
        10, 11, 12,
        13, 14, 15;

    // Matrices
    Matrix<float, m, n> Q = A;
    Matrix<float, n, n> R = Matrix<float, n, n>::Zero();
    Matrix<float, m, n> QtimesR = Matrix<float, m, n>::Zero();

    std::cout << "A = " << std::endl << A << std::endl << std::endl;

    // <T>LAPACK -----------------------------------------------
    
    std::cout << "--- <T>LAPACK: ---" << std::endl << std::endl;

    // Allocates memory
    Matrix<float, n, 1> tau;
    Matrix<float, n-1, 1> work;
    Matrix<float, n, n> orthQ;

    // Compute QR decomposision in place
    geqr2( Q, tau, work );
    // Copy the upper triangle to R
    lacpy( upperTriangle, slice(Q,pair{0,n},pair{0,n}), R );
    // Generate Q
    ung2r( n, Q, tau, work );

    std::cout << "Q = " << std::endl << Q << std::endl;
    std::cout << std::endl;

    std::cout << "R = " << std::endl << R << std::endl;
    std::cout << std::endl;

    // Checking A = Q R
    lacpy( dense, Q, QtimesR );
    trmm( Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, 1.0, R, QtimesR );
    std::cout << "QR = " << std::endl << QtimesR << std::endl;
    QtimesR -= A;
    std::cout << "\\|QR - A\\|_F/\\|A\\|_F = " << std::endl << lange( frob_norm, QtimesR ) / lange( frob_norm, A ) << std::endl;
    std::cout << std::endl;

    // Checking orthogonality of Q
    orthQ = Matrix<float, n, n>::Identity();
    syrk( Uplo::Upper, Op::Trans, 1.0, Q, -1.0, orthQ );
    std::cout << "\\|Q^t Q - I\\|_F = " << std::endl << lansy( frob_norm, upperTriangle, orthQ ) << std::endl;
    std::cout << std::endl;

    // Eigen -----------------------------------------------

    std::cout << "--- Eigen: ---" << std::endl << std::endl;

    // Compute QR decomposision in place, possibly allocating memory dynamically
    Eigen::HouseholderQR<decltype(A)> qrEigen( A );
    // Generate Q
    Q = qrEigen.householderQ() * Matrix<float, m, n>::Identity();
    // Copy the upper triangle to R
    R = qrEigen.matrixQR().block(0,0,n,n).triangularView<Eigen::Upper>();

    std::cout << "Q = " << std::endl << Q << std::endl;
    std::cout << std::endl;

    std::cout << "R = " << std::endl << R << std::endl;
    std::cout << std::endl;

    // Checking A = Q R
    QtimesR = Q * R;
    std::cout << "QR = " << std::endl << QtimesR << std::endl;
    std::cout << "\\|QR - A\\|_F/\\|A\\|_F = " << (QtimesR-A).norm() / A.norm() << std::endl;
    std::cout << std::endl;

    // Checking orthogonality of Q
    orthQ = Q.transpose() * Q - Matrix<float, n, n>::Identity();
    std::cout << "\\|Q^t Q - I\\|_F = " << std::endl << orthQ.norm() << std::endl;
    std::cout << std::endl;

    return 0;
}
