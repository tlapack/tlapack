/// @file test_getri.cpp
/// @brief Test functions that calculate inverse of matrices such as getri family.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/legacyArray.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>
#include "tlapack/lapack/getri_methodC.hpp"

using namespace tlapack;
using namespace std;

TEMPLATE_LIST_TEST_CASE("LU factorization of a general m-by-n matrix, blocked", "[lqf]", types_to_test)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t; // equivalent to using real_t = real_type<T>;
    
    //n represent no. rows and columns of the square matrices we will performing tests on
    idx_t n;
    n = GENERATE(5,10,20,100);

    // eps is the machine precision, and tol is the tolerance we accept for tests to pass
    const real_t eps = ulp<real_t>();
    const real_t tol = n*n*eps;
    
    // Initialize matrices A, and A_copy to run tests on
    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> A_copy_(new T[n * n]);
    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? n : n);
    auto A_copy = legacyMatrix<T, layout<matrix_t>>(n, n, &A_copy_[0], layout<matrix_t> == Layout::ColMajor ? n : n);
    
    // building identity matrix
    std::unique_ptr<T[]> ident1_(new T[n * n]);
    auto ident1 = legacyMatrix<T, layout<matrix_t>>(n, n, &ident1_[0], layout<matrix_t> == Layout::ColMajor ? n : n);
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i){
            if(i==j){
                ident1(i, j) = T(1);
            }
            else{
                ident1(i, j) = T(0);
            }
            
        }

    
    // forming A, a random matrix 
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i){
            A(i, j) = rand_helper<T>();
        }
            

    // make a deep copy A
    lacpy(Uplo::General, A, A_copy);
    
    // save norm of norma
    double norma=tlapack::lange( tlapack::Norm::Max, A);
    
    // run inverse function of choice
    getri_methodC(A);
    
    // identit1 <----- A * A_copy - ident1
    gemm(Op::NoTrans,Op::NoTrans,real_t(1),A,A_copy,real_t(-1),ident1);
    
       
    real_t error1 = tlapack::lange( tlapack::Norm::Max, ident1)/norma;
    CHECK(error1/tol <= 1);
    
}



