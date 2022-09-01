/// @file test_gelqf.cpp
/// @brief Test GELQF and UNGL2 and output a k-by-n orthogonal matrix Q.
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
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("LU factorization of a general m-by-n matrix, blocked", "[lqf]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t; // equivalent to using real_t = real_type<T>;

    //not sure if we need const 
    T zero(0);

    idx_t m, n;
    m = GENERATE(10, 20, 30);
    n = GENERATE(10, 20, 30);
    idx_t k=min<idx_t>(m,n);


    const real_t eps = ulp<real_t>();
    const real_t tol = max(m, n) * eps;

    std::unique_ptr<T[]> A_(new T[m * n]);
    std::unique_ptr<T[]> A_copy_(new T[m * n]);

    auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    auto A_copy = legacyMatrix<T, layout<matrix_t>>(m, n, &A_copy_[0], layout<matrix_t> == Layout::ColMajor ? m : n);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(Uplo::General, A, A_copy);
    
    double norma=tlapack::lange( tlapack::Norm::Max, A);

    getrf(A);
    std::vector<T> L_( m*k , T(0) );
    std::vector<T> U_( k*n , T(0) );

    auto L = legacyMatrix<T, layout<matrix_t>>(m, k, &L_[0], layout<matrix_t> == Layout::ColMajor ? m : k);
    auto U = legacyMatrix<T, layout<matrix_t>>(k, n, &U_[0], layout<matrix_t> == Layout::ColMajor ? k : n);
    for(idx_t i=0;i<m;i++){
        for(idx_t j=0;j<n;j++){
            if(i==j){
                L(i,j)=1.0;
                U(i,j)=A(i,j);
            }
            else if(i>j){
                L(i,j)=A(i,j);
            }
            else{
                U(i,j)=A(i,j);
            }
        }
    }
    gemm(Op::NoTrans,Op::NoTrans,real_t(1),L,U,real_t(-1),A_copy);

    real_t error = tlapack::lange( tlapack::Norm::Max, A_copy)/norma;
    CHECK(error <= tol);
}
//Proposoed modification
/*
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    //not sure if we need const 
    T zero(0);

    idx_t m, n;
    m = GENERATE(10, 20, 30);
    n = GENERATE(10, 20, 30);
    idx_t k=min<idx_t>(m,n);


    const real_t eps = ulp<real_t>();
    const real_t tol = max(m, n) * eps;

    std::unique_ptr<T[]> A_(new T[m * n]);
    std::unique_ptr<T[]> A_copy_(new T[m * n]);

    auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    auto A_copy = legacyMatrix<T, layout<matrix_t>>(m, n, &A_copy_[0], layout<matrix_t> == Layout::ColMajor ? m : n);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(Uplo::General, A, A_copy);
    
    double norma=tlapack::lange( tlapack::Norm::Max, A);

    LU(A);
    std::vector<T> L_( m*k , T(0) );
    std::vector<T> U_( k*n , T(0) );

    auto L = legacyMatrix<T, layout<matrix_t>>(m, k, &L_[0], layout<matrix_t> == Layout::ColMajor ? m : k);
    auto U = legacyMatrix<T, layout<matrix_t>>(k, n, &U_[0], layout<matrix_t> == Layout::ColMajor ? k : n);
    Question: How do I defualt L and U to be zero??
    for(idx_t i=0;i<m;i++){
        for(idx_t j=0;j<n;j++){
            if(i==j){
                L(i,j)=1.0;
                U(i,j)=A(i,j);
            }
            else if(i>j){
                L(i,j)=A(i,j);
            }
            else{
                U(i,j)=A(i,j);
            }
        }
    }
    gemm(Op::NoTrans,Op::NoTrans,1,L,U,-1,A_copy);

    real_t error = tlapack::lange( tlapack::Norm::Max, A_copy)/norma;
    CHECK(error <= tol);

*/
