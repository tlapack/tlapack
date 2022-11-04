/// @file test_ul_mult.cpp
/// @brief Test UL multiplication.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/ul_mult.hpp>

using namespace tlapack;
using namespace std;

TEMPLATE_LIST_TEST_CASE("LU factorization of a general m-by-n matrix, blocked", "[ul_mul]", types_to_test)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t; // equivalent to using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;
    
    // m and n represent no. rows and columns of the matrices we will be testing respectively
    idx_t n = GENERATE(5,10,20,30,100);

    // eps is the machine precision, and tol is the tolerance we accept for tests to pass
    const real_t eps = ulp<real_t>();
    const real_t tol = 10*n*eps;
    
    // Initialize matrices A, and A_copy to run tests on
    std::vector<T> A_; auto A = new_matrix( A_, n, n );
    std::vector<T> A_copy_; auto A_copy = new_matrix( A_copy_, n, n );

    // forming A, a random matrix with diagonal of 1
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i){
            if(i==j){
                A(i, j) = rand_helper<T>();

            }
            else{
                A(i, j) = rand_helper<T>();

            }
            
        }
            
    // We will make a deep copy A
    lacpy(Uplo::General, A, A_copy);
    double norma=tlapack::lange( tlapack::Norm::Max, A);
    
    // Put diagonal and super-diagonal of A into U and sub-diagonal in L  
    std::vector<T> L_; auto L = new_matrix( L_, n, n );
    std::vector<T> U_; auto U = new_matrix( U_, n, n );

    for (idx_t j = 0; j < n; ++j){
        for (idx_t i = 0; i < n; ++i){
            if(i==j){
                L(i, j) = T(1);
                U(i, j) = A(i,j);

            }
            else if (i>j)
            {
                L(i, j) = A(i,j);
                U(i, j) = T(0);
            }
            else{
                L(i, j) =T(0);
                U(i, j) = A(i,j);

            }
            
        }
    }

    // run ul_mult, which calculates U and L in place of A
    ul_mult(A);
    
    // store UL-A ---> A 
    gemm(Op::NoTrans,Op::NoTrans,T(1),U,L,T(-1),A);

    real_t error1 = tlapack::lange( tlapack::Norm::Max, A)/norma;
    CHECK(error1/tol <= 1);
    
}



