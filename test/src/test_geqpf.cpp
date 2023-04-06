/// @file test_geqpf.cpp
/// @author Racheal Asamoah, University of Colorado Denver, USA
/// @brief Test GEQPF and UNG2R
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/geqpf.hpp>
#include <tlapack/lapack/laqps.hpp>
#include <tlapack/lapack/ung2r.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("QR factorization with column pivoting of a general m-by-n matrix",
                   "[qpf]",
                   TLAPACK_TYPES_TO_TEST
                //    legacyMatrix<double>
                //    legacyMatrix<std::complex<float>>
                   )
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    idx_t m, n, k;

    // m = 19;
    // n = 9;
    m = GENERATE(9, 19, 30);
    n = GENERATE(9, 19, 30);
    k = std::min<idx_t>(m,n);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(2 * std::max<idx_t>(m,n)) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, m);

    std::vector<idx_t> jpvt(k);
    std::vector<T> tauw(k);

    // Workspace computation:
    workinfo_t workinfo = {};
    geqpf_worksize(A, jpvt, tauw, workinfo);
    ung2r_worksize(Q, tauw, workinfo);

    // Workspace allocation:
    vectorOfBytes workVec;
    workspace_opts_t<> workOpts(alloc_workspace(workVec, workinfo));

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(Uplo::General, A, A_copy);

    // if (k > n) return;

    INFO("m = " << m << " n = " << n);
    {
        // geqpf(A, jpvt, tauw, workOpts);
        laqp3( A, jpvt, tauw, workOpts );

        auto Q0 = slice(Q, range(0, m), range(0, k));
        lacpy(Uplo::General, slice(A, range(0, m), range(0, k)), Q0);

        ung2r(Q, tauw, workOpts);

        auto orth_Q = check_orthogonality(Q);
        // std::cout << orth_Q << "\n";
        CHECK(orth_Q <= tol);

        // check that A * P = Q * R
        // compute A_copy = A * P - Q * R

        // firstly, permute columns of A_copy to form in A * P in A_copy
        for (idx_t j = 0; j != k; j++) {
            auto vect1 = tlapack::col(A_copy, j);
            auto vect2 = tlapack::col(A_copy, jpvt[j]);
            tlapack::swap(vect1, vect2);
        }

        // secondly, if m < n, compute columns from m to n-1
        auto A1 = slice(A_copy, range(0, m), (k < n) ? range(k, n) : range(0,0));
        auto R1 = slice(A, range(0, k), (k < n) ? range(k, n) : range(0,0));
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), Q0, R1, real_t(-1.), A1);
        
        // thirdly, in all cases, consider columns from 0 to k
        auto R0 = slice(A, range(0, k), range(0, k));
        trmm(Side::Right, Uplo::Upper,Op::NoTrans,Diag::NonUnit, real_t(1.), R0, Q0);
        for (idx_t i = 0; i < m; ++i) {
            for(idx_t j = 0; j < k; ++j) {
                A_copy(i,j) = A_copy(i,j)-Q0(i,j);
            }
        }

        real_t repres =
            lange(Norm::Max, slice(A_copy, range(0, m), range(0, n)));
        // std::cout << repres << "\n";

        CHECK(repres <= tol);


        // Check diagonal of R is nonincreasing (in modulus)
        bool diagonal_is_nonincreasing = true;
        for (idx_t i = 0; i < k-1; ++i) { 
            diagonal_is_nonincreasing = diagonal_is_nonincreasing 
            && (tlapack::abs(R0(i,i))*(1+tol) >= tlapack::abs(R0(i+1,i+1))); 
        }
        for (idx_t i = 0; i < k; ++i) { 
            UNSCOPED_INFO(tlapack::abs(R0(i,i)));
        }
        // std::cout << diagonal_is_nonincreasing << "\n";
        CHECK( diagonal_is_nonincreasing ); 
    
        // for (idx_t i = 0; i < k; ++i) { 
        //     std::cout << tlapack::abs(R0(i,i)) << "\n";
        // }


    }
}
