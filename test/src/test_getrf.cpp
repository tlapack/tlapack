/// @file test_getrf.cpp
/// @brief Test the LU factorization of a matrix A
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "testutils.hpp"
#include <tlapack.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("LU factorization of a general m-by-n matrix", "[getrf]", types_to_test)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t; // equivalent to using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // m and n represent no. rows and columns of the matrices we will be testing respectively
    idx_t m = GENERATE(10, 20, 30);
    idx_t n = GENERATE(10, 20, 30);
    GetrfVariant variant = GENERATE( GetrfVariant::Level0, GetrfVariant::Recursive );
    
    idx_t k = min<idx_t>(m,n);

    // eps is the machine precision, and tol is the tolerance we accept for tests to pass
    const real_t eps = ulp<real_t>();
    const real_t tol = max(m, n) * eps;

    // Initialize matrices A, and A_copy to run tests on
    std::vector<T> A_; auto A = new_matrix( A_, m, n );
    std::vector<T> A_copy_; auto A_copy = new_matrix( A_copy_, m, n );

    // Update A with random numbers
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i){
            // A(i, j) = rand_helper<T>();
            A(i, j) = rand_helper<T>();
        }

    // We will make a deep copy A
    // We intend to test A=LU, however, since after calling getrf, A will be udpated
    // then to test A=LU, we'll make a deep copy of A prior to calling getrf
    lacpy(Uplo::General, A, A_copy);

    double norma=tlapack::lange( tlapack::Norm::Max, A);
    // Initialize Piv vector to all zeros
    std::vector<idx_t> Piv( k , idx_t(0) );
    // Run getrf and both A and Piv will be update
    getrf(A,Piv, getrf_opts_t{variant} );

    // A contains L and U now, then form A <--- LU
    if( m > n )
    {
        auto A0 = tlapack::slice(A,tlapack::range<idx_t>(0,n),tlapack::range<idx_t>(0,n));
        auto A1 = tlapack::slice(A,tlapack::range<idx_t>(n,m),tlapack::range<idx_t>(0,n));
        trmm( Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, real_t(1), A0, A1 );
        lu_mult( A0 );
    }
    else if( m < n )
    {
        auto A0 = tlapack::slice(A,tlapack::range<idx_t>(0,m),tlapack::range<idx_t>(0,m));
        auto A1 = tlapack::slice(A,tlapack::range<idx_t>(0,m),tlapack::range<idx_t>(m,n));
        trmm( Side::Left, Uplo::Lower, Op::NoTrans, Diag::Unit, real_t(1), A0, A1 );
        lu_mult( A0 );
    }
    else
        lu_mult( A );

    // Now that Piv is updated, we work our way backwards in Piv and switch rows of LU
    for(idx_t j=k-idx_t(1);j!=idx_t(-1);j--){
        auto vect1=tlapack::row(A,j);
        auto vect2=tlapack::row(A,Piv[j]);
        tlapack::swap(vect1,vect2);
    }

    // A <- A_original - LU
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < n; j++)
            A(i,j) -= A_copy(i,j);

    // Check for relative error: norm(A-LU)/norm(A)
    real_t error = tlapack::lange( tlapack::Norm::Max, A)/norma;
    CHECK(error <= tol);

}
