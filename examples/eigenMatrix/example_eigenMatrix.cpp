/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#define NDEBUG 1

#include <iostream>
#include <plugins/tlapack_eigen.hpp>
#include <tlapack.hpp>

#include <Eigen/Householder>

int main( int argc, char** argv )
{
    using std::size_t;
    using pair = std::pair<size_t,size_t>;
    using blas::submatrix;

    const size_t m = 5;
    const size_t n = 3;
    const size_t k = (m <= n) ? m : n;

    Eigen::Matrix<float, m, n> A {
        { 1,  2,  3},
        { 4,  5,  6},
        { 7,  8,  9},
        {10, 11, 12},
        {13, 14, 15}
    };
    Eigen::Matrix<float, m, n> Q = A;

    Eigen::Matrix<float, n, n> R = Eigen::Matrix<float, n, n>::Zero();
    Eigen::Matrix<float, m, n> QtimesR = Eigen::Matrix<float, m, n>::Zero();

    std::cout << "A = " << std::endl << A << std::endl << std::endl;

    // LAPACK
    std::cout << "--- <T>LAPACK: ---" << std::endl << std::endl;

    Eigen::Matrix<float, k, 1> tau;
    Eigen::Matrix<float, n-1, 1> work;
    Eigen::Matrix<float, n, n> orthQ = Eigen::Matrix<float, n, n>::Identity();

    lapack::geqr2( Q, tau, work );
    lapack::lacpy(  lapack::upper_triangle, 
                    submatrix(Q,pair(0,k),pair(0,k)), 
                    R );
    lapack::org2r( k, Q, tau, work );

    std::cout << "Q = " << std::endl << Q << std::endl;
    std::cout << std::endl;

    std::cout << "R = " << std::endl << R << std::endl;
    std::cout << std::endl;

    blas::gemm( blas::Op::NoTrans, blas::Op::NoTrans,
        1.0, Q, R, 0.0, QtimesR);
    std::cout << "Q*R = " << std::endl << QtimesR << std::endl;
    std::cout << std::endl;

    blas::syrk( blas::Uplo::Upper, blas::Op::Trans, 1.0, Q, -1.0, orthQ);
    std::cout << "\\|Q^t Q\\|_F = " << std::endl << lapack::lansy( lapack::frob_norm, lapack::upper_triangle, orthQ ) << std::endl;
    std::cout << std::endl;

    // Eigen
    std::cout << "--- Eigen: ---" << std::endl << std::endl;

    Eigen::HouseholderQR<decltype(A)> qrEigen( A );

    Q = qrEigen.householderQ() * Eigen::MatrixXf::Identity(5,3);
    R = qrEigen.matrixQR().block(0,0,n,n).triangularView<Eigen::Upper>();

    std::cout << "Q = " << std::endl << Q << std::endl;
    std::cout << std::endl;

    std::cout << "R = " << std::endl << R << std::endl;
    std::cout << std::endl;

    QtimesR = Q * R;
    std::cout << "Q*R = " << std::endl << QtimesR << std::endl;
    std::cout << std::endl;

    orthQ = Q.transpose() * Q - Eigen::MatrixXf::Identity(n,n);
    std::cout << "\\|Q^t Q\\|_F = " << std::endl << orthQ.norm() << std::endl;
    std::cout << std::endl;

    return 0;
}
