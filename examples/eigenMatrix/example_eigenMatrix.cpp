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
    Eigen::Matrix<float, n, n> R = Eigen::Matrix<float, n, n>::Zero();
    Eigen::Matrix<float, k, 1> tau;
    Eigen::Matrix<float, n-1, 1> work;
    Eigen::Matrix<float, m, n> QtimesR = Eigen::Matrix<float, m, n>::Zero();

    std::cout << A << std::endl;

    lapack::geqr2( A, tau, work );
    lapack::lacpy(  lapack::upper_triangle, 
                    submatrix(A,pair(0,k),pair(0,k)), 
                    R );
    lapack::org2r( k, A, tau, work );

    std::cout << A << std::endl;
    std::cout << R << std::endl;

    blas::gemm( blas::Op::NoTrans, blas::Op::NoTrans,
        1.0, A, R, 0.0, QtimesR);

    std::cout << QtimesR << std::endl;

    return 0;
}
