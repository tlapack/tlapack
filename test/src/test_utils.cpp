/// @file test_utils.cpp
/// @brief Test utils from <T>LAPACK.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>

#include "testdefinitions.hpp"

using namespace tlapack;

TEST_CASE( "Random generator is consistent if seed is fixed", "[utils]" ) {
    rand_generator gen;
    gen.seed(6845315);

    CHECK( gen() == 1225581775 );
    CHECK( gen() == 1985311242 );
    CHECK( gen() == 300629471 );
    CHECK( gen() == 2636314308 );
    CHECK( gen() == 1603395911 );
    CHECK( gen() == 393807335 );
    CHECK( gen() == 3641191292 );
}

TEST_CASE( "MatrixAccessPolicy can be cast to Uplo", "[utils]" ) {
    Uplo uplo;
    MatrixAccessPolicy runtimeAccess;
    
    uplo = Uplo::Upper;
    runtimeAccess = MatrixAccessPolicy::UpperTriangle;
    
    CHECK( uplo == (Uplo) runtimeAccess );
    CHECK( uplo == upperTriangle );
    CHECK( uplo == (Uplo) upperTriangle );

    CHECK( runtimeAccess == (MatrixAccessPolicy) uplo );
    CHECK( runtimeAccess == upperTriangle );
    CHECK( runtimeAccess == (MatrixAccessPolicy) upperTriangle );

    CHECK( upperTriangle == (Uplo) runtimeAccess );
    CHECK( upperTriangle == (MatrixAccessPolicy) uplo );

    uplo = Uplo::Lower;
    runtimeAccess = MatrixAccessPolicy::LowerTriangle;
    
    CHECK( uplo == (Uplo) runtimeAccess );
    CHECK( uplo == lowerTriangle );
    CHECK( uplo == (Uplo) lowerTriangle );

    CHECK( runtimeAccess == (MatrixAccessPolicy) uplo );
    CHECK( runtimeAccess == lowerTriangle );
    CHECK( runtimeAccess == (MatrixAccessPolicy) lowerTriangle );

    CHECK( lowerTriangle == (Uplo) runtimeAccess );
    CHECK( lowerTriangle == (MatrixAccessPolicy) uplo );

    uplo = Uplo::General;
    runtimeAccess = MatrixAccessPolicy::Dense;
    
    CHECK( uplo == (Uplo) runtimeAccess );
    CHECK( uplo == dense );
    CHECK( uplo == (Uplo) dense );

    CHECK( runtimeAccess == (MatrixAccessPolicy) uplo );
    CHECK( runtimeAccess == dense );
    CHECK( runtimeAccess == (MatrixAccessPolicy) dense );

    CHECK( dense == (Uplo) runtimeAccess );
    CHECK( dense == (MatrixAccessPolicy) uplo );
}

TEMPLATE_LIST_TEST_CASE("is_matrix works", "[utils]", types_to_test)
{
    using matrix_t  = TestType;

    CHECK( is_matrix<matrix_t> );
}

TEST_CASE("is_matrix and is_vector work", "[utils]")
{
    CHECK( !is_matrix< std::vector<float> > );
    CHECK( !is_matrix< legacyVector<float> > );

    CHECK( is_vector< std::vector<float> > );
    CHECK( is_vector< legacyVector<float> > );

    CHECK( !is_matrix< float > );
    CHECK( !is_matrix< std::complex<double> > );

    CHECK( !is_vector< float > );
    CHECK( !is_vector< std::complex<double> > );
}