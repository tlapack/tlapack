/// @file test_getri.cpp
/// @brief Test functions that calculate inverse of matrices such as getri family.
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("Inversion of a general m-by-n matrix", "[getri]", types_to_test)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t; // equivalent to using real_t = real_type<T>;
    
    //n represent no. rows and columns of the square matrices we will performing tests on
    idx_t n = GENERATE(5,10,20,100);
    GetriVariant variant = GENERATE( GetriVariant::UXLI, GetriVariant::UILI );

    // eps is the machine precision, and tol is the tolerance we accept for tests to pass
    const real_t eps = ulp<real_t>();
    const real_t tol = n*n*eps;
    
    // Initialize matrices A, and invA to run tests on
    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> invA_(new T[n * n]);
    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &A_[0], n);
    auto invA = legacyMatrix<T, layout<matrix_t>>(n, n, &invA_[0], n);
    
    // forming A, a random matrix 
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i){
            A(i, j) = rand_helper<T>();
        }

    
    // make a deep copy A
    lacpy(Uplo::General, A, invA);
    
    // calculate norm of A for later use in relative error
    double norma=tlapack::lange( tlapack::Norm::Max, A);
    
    // LU factorize Pivoted A
    std::vector<idx_t> Piv( n , idx_t(0) );
    getrf(invA,Piv);

    // run inverse function, this could test any inverse function of choice
    std::vector<T> work( n , T(0) );
    getri_opts_t< std::vector<T> > opts = { variant, &work };
    getri(invA,Piv,opts);

    // building error matrix E
    std::unique_ptr<T[]> E_(new T[n * n]);
    legacyMatrix<T, layout<matrix_t>> E(n, n, &E_[0], n);
    
    // E <----- inv(A)*A - I
    gemm(Op::NoTrans,Op::NoTrans,real_t(1),A,invA,E);
    for (idx_t i = 0; i < n; i++)
        E(i,i) -= real_t(1);
    
    // error is  || inv(A)*A - I || / ( ||A|| * ||inv(A)|| )
    real_t error = tlapack::lange( tlapack::Norm::Max, E)
                    / (norma * tlapack::lange( tlapack::Norm::Max, invA ));

    INFO( "|| inv(A)*A - I || / ( ||A|| * ||inv(A)|| )" );
    INFO( "n = " << n );
    INFO( "variant = " << (char) variant );
    CHECK(error/tol <= 1); // tests if error<=tol
    
    // E <----- A*inv(A) - I
    gemm(Op::NoTrans,Op::NoTrans,real_t(1),invA,A,E);
    for (idx_t i = 0; i < n; i++)
        E(i,i) -= real_t(1);
    
    // error is  || A*inv(A) - I || / ( ||A|| * ||inv(A)|| )
    error = tlapack::lange( tlapack::Norm::Max, E)
                    / (norma * tlapack::lange( tlapack::Norm::Max, invA ));

    INFO( "|| A*inv(A) - I || / ( ||A|| * ||inv(A)|| )" );
    INFO( "n = " << n );
    INFO( "variant = " << (char) variant );
    CHECK(error/tol <= 1); // tests if error<=tol
    
}



