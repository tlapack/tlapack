/// @file test_eigenplugin.cpp
/// @brief Test plugin for integration with Eigen.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <tlapack/plugins/eigen.hpp>
#include <tlapack/base/utils.hpp>

TEST_CASE("is_eigen_type and is_eigen_map work", "[plugins]")
{
    // Non-class types
    CHECK( tlapack::internal::is_eigen_type< int > == false );
    CHECK( tlapack::internal::is_eigen_type< float > == false );

    // Class types
    CHECK( tlapack::internal::is_eigen_type< std::complex<float> > == false );
    CHECK( tlapack::internal::is_eigen_type< std::string > == false );

    // Eigen types
    CHECK( tlapack::internal::is_eigen_type< Eigen::MatrixXd > == true );
    CHECK( tlapack::internal::is_eigen_type< Eigen::Matrix<float,2,2> > == true );
    CHECK( tlapack::internal::is_eigen_type< Eigen::Matrix<float,3,10> > == true );
    CHECK( tlapack::internal::is_eigen_type< Eigen::Block<Eigen::MatrixXd> > == true );
    CHECK( tlapack::internal::is_eigen_type< Eigen::Map< Eigen::MatrixXd > > == true );
    
    // // Non-class types
    // CHECK( tlapack::internal::is_eigen_map< int > == false );
    // CHECK( tlapack::internal::is_eigen_map< float > == false );

    // // Class types
    // CHECK( tlapack::internal::is_eigen_map< std::complex<float> > == false );
    // CHECK( tlapack::internal::is_eigen_map< std::string > == false );

    // // Eigen types
    // CHECK( tlapack::internal::is_eigen_map< Eigen::MatrixXd > == false );
    // CHECK( tlapack::internal::is_eigen_map< Eigen::Matrix2<float> > == false );
    // CHECK( tlapack::internal::is_eigen_map< Eigen::Matrix<float,3,10> > == false );
    // CHECK( tlapack::internal::is_eigen_map< Eigen::Block<Eigen::MatrixXd> > == false );
    
    // CHECK( tlapack::internal::is_eigen_map< Eigen::Map< Eigen::MatrixXd > > == true );
    // CHECK( tlapack::internal::is_eigen_map< Eigen::MapBase< Eigen::Map< Eigen::MatrixXd > > > == false );

    // Eigen::Map< Eigen::MatrixXd > map1(NULL,0,0);
    // Eigen::MapBase< Eigen::Map< Eigen::MatrixXd > > map2( NULL, 0, 0 );
    
    // CHECK( tlapack::internal::is_eigen_map_f(&map1) == true );
    // CHECK( tlapack::internal::is_eigen_map_f(&map2) == false );

    CHECK( tlapack::internal::is_eigen_block< Eigen::Block<Eigen::MatrixXd> > == true );
    CHECK( tlapack::internal::is_eigen_block< Eigen::Block<Eigen::Matrix<float,-1,-1,1,-1,-1>> > == true );

    CHECK( tlapack::internal::is_eigen_block< Eigen::Matrix<float,-1,-1,1,-1,-1> > == false );
}

TEST_CASE("legacy_matrix works", "[plugins]")
{
    {
        Eigen::Matrix2d A;
        A << 1, 3,
             2, 4;
        Eigen::MatrixXd A2 = A.block<1,2>(1,0);

        auto B = tlapack::legacy_matrix( A );
        CHECK( B(0,0) == A(0,0) );
        CHECK( B(1,0) == A(1,0) );
        CHECK( B(0,1) == A(0,1) );
        CHECK( B(1,1) == A(1,1) );

        auto B2 = tlapack::legacy_matrix( A2 );
        CHECK( B2(0,0) == 2 );
        CHECK( B2(0,1) == 4 );

        auto B3 = tlapack::legacy_vector( A2 );
        CHECK( B3[0] == 2 );
        CHECK( B3[1] == 4 );
    }
    {
        Eigen::Matrix<float,-1,-1,Eigen::RowMajor> A( 3, 5 );
        A << 1, 2, 3, 4, 5,
            6, 7, 8, 9, 0,
            -1, -2, -3, -4, -5;

        auto B = tlapack::legacy_matrix( A );
        CHECK( B(0,0) == A(0,0) );
        CHECK( B(2,4) == A(2,4) );

        Eigen::Matrix<float,-1,1> A2 = A.col(3);
        auto B2 = tlapack::legacy_vector( A2 );
        CHECK( B2[0] == A(0,3) );
        CHECK( B2[1] == A(1,3) );
    }
}

TEST_CASE("slice works", "[plugins]")
{
    using pair = tlapack::range<Eigen::Index>;

    {
        Eigen::Matrix2d A;
        A << 1, 3,
            2, 4;

        auto B = tlapack::slice( A, pair{1,2}, pair{1,2} );
        CHECK( B(0,0) == A(1,1) );
    }
    {
        Eigen::MatrixXd A(2,2);
        A << 1, 3,
             2, 4;

        auto B = tlapack::slice( A, pair{1,2}, pair{1,2} );
        CHECK( B(0,0) == A(1,1) );
    }
}

TEST_CASE("Layout works", "[plugins]")
{
    CHECK( tlapack::layout< Eigen::Matrix<float,-1,-1,0,-1,-1> > == tlapack::Layout::ColMajor );
    CHECK( tlapack::layout< Eigen::Matrix<float,-1,-1,1,-1,-1> > == tlapack::Layout::RowMajor );

    CHECK( tlapack::layout< Eigen::Block<Eigen::MatrixXd> > == tlapack::Layout::ColMajor );

    CHECK( tlapack::layout< Eigen::Block<Eigen::Matrix<float,-1,-1,1,-1,-1>> > == tlapack::Layout::RowMajor );

    CHECK( tlapack::layout< Eigen::Block<Eigen::Matrix<float,-1,-1,1,-1,-1>,-1,1> > == tlapack::Layout::RowMajor );
    CHECK( tlapack::layout< Eigen::Block<Eigen::Matrix<float,-1,-1,1,-1,-1>,1,-1> > == tlapack::Layout::RowMajor );
    CHECK( tlapack::layout< Eigen::Block<Eigen::Matrix<float,-1,-1,0,-1,-1>,-1,1> > == tlapack::Layout::ColMajor );
    CHECK( tlapack::layout< Eigen::Block<Eigen::Matrix<float,-1,-1,0,-1,-1>,1,-1> > == tlapack::Layout::ColMajor );

    {
        using Derived = Eigen::Block<Eigen::Matrix<float, -1, -1, 1, -1, -1>, -1, 1, false>;

        CHECK( tlapack::layout< Derived > == tlapack::Layout::RowMajor );
    }
}