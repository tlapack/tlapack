/// @file test_getri.cpp
/// @brief Test functions that calculate inverse of matrices such as getri family.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>


#include "testutils.hpp"
#include <tlapack.hpp>
#include "tlapack/base/legacyArray.hpp"
#include "tlapack/lapack/geqr2.hpp"


using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("Inversion of a general m-by-n matrix", "[getri]", types_to_test)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t; // equivalent to using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;
    
    //n represent no. rows and columns of the square matrices we will performing tests on
    // idx_t m = GENERATE(21,20, 10);
    // idx_t n = GENERATE(13,10,5);
    idx_t n = GENERATE(21,20, 10);
    idx_t m = GENERATE(10,5);
    idx_t k = min<idx_t>(m,n);

    
    // eps is the machine precision, and tol is the tolerance we accept for tests to pass
    const real_t eps = ulp<real_t>();
    const real_t tol = m*m*m*m*n*m*n*eps;

    
    std::vector<T> tau(k);
    for (idx_t i = 0; i < k; ++i){
        tau[i]=rand_helper<T>();
    }
    std::vector<T> C_; auto C = new_matrix( C_, m, n );
    
    
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i){
            if(j-i==n-k){
                C(i, j) = T(1);
            }
            else if (j-i>n-k)
            {
                C(i, j) = T(0);
            }
            else
            {
                C(i, j) = rand_helper<T>();
            }
        }

    real_t normc;
    normc=tlapack::lange( tlapack::Norm::Max, C);
    //tlapack::geqr2( C, tau );

    std::vector<T> TT_; auto TT = new_matrix( TT_, k, k );
    std::vector<T> TTT_; auto TTT = new_matrix( TTT_, k, k );
    for (idx_t i = 0; i < k; ++i)
        for (idx_t j = 0; j < k; ++j){
            TT(i,j)=T(0);
            TTT(i,j)=T(0);
        }
        
            
    
    
    tlapack::larft(tlapack::Direction::Backward, tlapack::rowwise_storage, C, tau, TT);
    larft_recursive(tlapack::Direction::Backward,tlapack::StoreV::Rowwise,C, tau, TTT);
    
    // for (idx_t i = 0; i < k; ++i)
    //     for (idx_t j = 0; j < k; ++j){
    //         if(i<=j)
    //         std::cout<<TTT(i,j);
    //     }
    for (idx_t i = 0; i < k; ++i)
        for (idx_t j = 0; j < k; ++j){
            TTT(i,j)-=TT(i,j);
        }
    

    real_t error1;
    error1 = tlapack::lange( tlapack::Norm::Max, TTT)
                    / (normc);

    
    CHECK(error1 /tol <=1); // tests if error<=tol
    
}



