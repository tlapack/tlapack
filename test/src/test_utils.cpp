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

using namespace tlapack;

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
