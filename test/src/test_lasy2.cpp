/// @file test_lasy2.cpp
/// @brief Test 1x1 and 2x2 sylvester solver
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "testutils.hpp"

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/lasy2.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("sylvester solver gives correct result", "[sylvester]", TLAPACK_REAL_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    const T one(1);
    idx_t n1 = GENERATE(1, 2);
    // Once 1x2 solver is finished, generate n2 independantly
    idx_t n2 = n1;
    const real_t eps = uroundoff<real_t>();
    const real_t tol = real_t(1.0e2f) * eps;

    std::vector<T> TL_; auto TL = new_matrix( TL_, n1, n1 );
    std::vector<T> TR_; auto TR = new_matrix( TR_, n2, n2 );
    std::vector<T> B_; auto B = new_matrix( B_, n1, n2 );
    std::vector<T> X_; auto X = new_matrix( X_, n1, n2 );
    std::vector<T> X_exact_; auto X_exact = new_matrix( X_exact_, n1, n2 );

    for (idx_t i = 0; i < n1; ++i)
        for (idx_t j = 0; j < n1; ++j)
            TL(i, j) = rand_helper<T>();

    for (idx_t i = 0; i < n2; ++i)
        for (idx_t j = 0; j < n2; ++j)
            TR(i, j) = rand_helper<T>();

    for (idx_t i = 0; i < n1; ++i)
        for (idx_t j = 0; j < n2; ++j)
            X_exact(i, j) = rand_helper<T>();

    Op trans_l = Op::NoTrans;
    Op trans_r = Op::NoTrans;

    real_t sign( 1 );

    // Calculate op(TL)*X + ISGN*X*op(TR)
    gemm(trans_l, Op::NoTrans, one, TL, X_exact, B);
    gemm(Op::NoTrans, trans_r, sign, X_exact, TR, one, B);


    INFO("n1 = " << n1 << " n2 =" << n2);
    {
        // Solve sylvester equation
        T scale(0), xnorm;
        lasy2(Op::NoTrans, Op::NoTrans, 1, TL, TR, B, scale, X, xnorm);

        // Check that X_exact == X
        for (idx_t i = 0; i < n1; ++i)
            for (idx_t j = 0; j < n2; ++j)
                CHECK(abs1(X_exact(i, j) - scale * X(i, j)) <= tol * X_exact(i, j));
    }
}
