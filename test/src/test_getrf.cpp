/// @file test_getrf.cpp
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
using namespace std;

TEMPLATE_LIST_TEST_CASE("LU factorization of a general m-by-n matrix, blocked", "[lqf]", types_to_test)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t; // equivalent to using real_t = real_type<T>;
    
    // m and n represent no. rows and columns of the matrices we will be testing respectively
    idx_t m, n;
    m = GENERATE(10, 20, 30);
    n = GENERATE(10, 20, 30);
    idx_t k=min<idx_t>(m,n);

    // eps is the machine precision, and tol is the tolerance we accept for tests to pass
    const real_t eps = ulp<real_t>();
    const real_t tol = max(m, n) * eps;
    
    // Initialize matrices A, and A_copy to run tests on
    std::unique_ptr<T[]> A_(new T[m * n]);
    std::unique_ptr<T[]> A_copy_(new T[m * n]);
    auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    auto A_copy = legacyMatrix<T, layout<matrix_t>>(m, n, &A_copy_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    
    // Update A with random numbers
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    
    // We will make a deep copy A
    // We intend to test A=LU, however, since after calling getrf, A will be udpated
    // then to test A=LU, we'll make a deep copy of A prior to calling getrf
    lacpy(Uplo::General, A, A_copy);
    
    double norma=tlapack::lange( tlapack::Norm::Max, A);
    // Initialize Piv vector to all zeros
    std::vector<idx_t> Piv( k , idx_t(0) );
    // Run getrf and both A and Piv will be update
    getrf(A,Piv);
    // Initialize L and U
    std::vector<T> L_( m*k , T(0) );
    std::vector<T> U_( k*n , T(0) );
    auto L = legacyMatrix<T, layout<matrix_t>>(m, k, &L_[0], layout<matrix_t> == Layout::ColMajor ? m : k);
    auto U = legacyMatrix<T, layout<matrix_t>>(k, n, &U_[0], layout<matrix_t> == Layout::ColMajor ? k : n);
    
    // construct L and U in one pass from the matrix A
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
    
    // Now that Piv is updated, we work our way backwards in Piv and switch rows of L
    for(idx_t j=k-idx_t(1);j!=idx_t(-1);j--){
        auto vect1=tlapack::row(L,j);
        auto vect2=tlapack::row(L,Piv[j]);
        tlapack::swap(vect1,vect2);
    }
    
    // Now that L is updated, we test if LU=A_copy
    gemm(Op::NoTrans,Op::NoTrans,real_t(1),L,U,real_t(-1),A_copy);
    real_t error = tlapack::lange( tlapack::Norm::Max, A_copy)/norma;
    CHECK(error <= tol);
    
}