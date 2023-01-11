/// @file test_gelqf.cpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @brief Test GELQF and UNGL2 and output a k-by-n orthogonal matrix Q.
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
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
#include <tlapack/lapack/gelqf.hpp>
#include <tlapack/lapack/ungl2.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("LQ factorization of a general m-by-n matrix, blocked", "[lqf]", TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    const T zero(0);

    idx_t m, n, k, nb;

    m = GENERATE(10, 20, 30);
    n = GENERATE(10, 20, 30);
    k = GENERATE(8, 10, 20, 30); // k is the number of rows for output Q. Can personalize it.
    nb = GENERATE(2, 3, 7, 12);  // nb is the block height. Can personalize it.

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(m*n) * eps;

    std::vector<T> A_; auto A = new_matrix( A_, m, n );
    std::vector<T> A_copy_; auto A_copy = new_matrix( A_copy_, m, n );
    std::vector<T> TT_; auto TT = new_matrix( TT_, m, nb );
    std::vector<T> Q_; auto Q = new_matrix( Q_, k, n );

    std::vector<T> tauw(min(m, n));

    // Workspace computation:
    gelqf_opts_t<idx_t> workOpts; workOpts.nb = nb;
    workinfo_t workinfo;
    gelqf_worksize(A, TT, workinfo, workOpts);
    ungl2_worksize(Q, tauw, workinfo, workOpts);

    // Workspace allocation:
    vectorOfBytes workVec;
    workOpts.work = alloc_workspace( workVec, workinfo );

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(Uplo::General, A, A_copy);

    if (k <= n) // k must be less than or equal to n, because we cannot get a Q bigger than n-by-n
    {
        INFO("m = " << m << " n = " << n << " k = " << k << " nb = " << nb);
        {
            gelqf(A, TT, workOpts);

            // Build tauw vector from matrix TT
            for (idx_t j = 0; j < min(m,n); j += nb)
            {
                idx_t ib = std::min<idx_t>(nb, min(m,n) - j);
                
                for (idx_t i = 0; i < ib; i++)
                    tauw[i+j] = TT(i+j,i);
            }

            // Q is sliced down to the desired size of output Q (k-by-n).
            // It stores the desired number of Householder reflectors that UNGL2 will use.
            lacpy(Uplo::General, slice(A, range(0, min(m, k)), range(0, n)), Q);

            ungl2(Q, tauw, workOpts);

            // Wq is the identity matrix to check the orthogonality of Q
            std::vector<T> Wq_; auto Wq = new_matrix( Wq_, k, k );
            auto orth_Q = check_orthogonality(Q, Wq);
            CHECK(orth_Q <= tol);

            // L is sliced from A after GELQ2
            std::vector<T> L_; auto L = new_matrix( L_, min(k, m), k );
            laset(Uplo::Upper, zero, zero, L);
            lacpy(Uplo::Lower, slice(A, range(0, min(m, k)), range(0, k)), L);

            // R stores the product of L and Q
            std::vector<T> R_; auto R = new_matrix( R_, min(k, m), n );

            // Test A = L * Q
            gemm(Op::NoTrans, Op::NoTrans, real_t(1.), L, Q, R);
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < min(m, k); ++i)
                    A_copy(i, j) -= R(i, j);

            real_t repres = tlapack::lange(tlapack::Norm::Max, slice(A_copy, range(0, min(m, k)), range(0, n)));
            CHECK(repres <= tol);

        }
    }
}
