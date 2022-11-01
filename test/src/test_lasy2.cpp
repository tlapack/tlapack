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
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("sylvester solver gives correct result", "[sylvester]", real_types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    const T one(1);
    idx_t n1 = GENERATE(1, 2);
    // Once 1x2 solver is finished, generate n2 independantly
    idx_t n2 = n1;
    const real_t eps = uroundoff<real_t>();
    const real_t tol = 1.0e2 * eps;

    std::unique_ptr<T[]> TL_(new T[n1 * n1]);
    std::unique_ptr<T[]> TR_(new T[n2 * n2]);
    std::unique_ptr<T[]> B_(new T[n1 * n2]);
    std::unique_ptr<T[]> X_(new T[n1 * n2]);
    std::unique_ptr<T[]> X_exact_(new T[n1 * n2]);

    auto TL = legacyMatrix<T, idx_t, layout<matrix_t>>(n1, n1, &TL_[0], n1);
    auto TR = legacyMatrix<T, idx_t, layout<matrix_t>>(n2, n2, &TR_[0], n2);
    auto B = legacyMatrix<T, idx_t, layout<matrix_t>>(n1, n2, &B_[0], layout<matrix_t> == Layout::ColMajor ? n1 : n2);
    auto X = legacyMatrix<T, idx_t, layout<matrix_t>>(n1, n2, &X_[0], layout<matrix_t> == Layout::ColMajor ? n1 : n2);
    auto X_exact = legacyMatrix<T, idx_t, layout<matrix_t>>(n1, n2, &X_exact_[0], layout<matrix_t> == Layout::ColMajor ? n1 : n2);

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

    int sign = 1;

    // Calculate op(TL)*X + ISGN*X*op(TR)
    gemm(trans_l, Op::NoTrans, one, TL, X_exact, B);
    gemm(Op::NoTrans, trans_r, sign, X_exact, TR, one, B);


    DYNAMIC_SECTION("n1 = " << n1 << " n2 =" << n2)
    {
        // Solve sylvester equation
        T scale, xnorm;
        lasy2(Op::NoTrans, Op::NoTrans, 1, TL, TR, B, scale, X, xnorm);

        // Check that X_exact == X
        for (idx_t i = 0; i < n1; ++i)
            for (idx_t j = 0; j < n2; ++j)
                CHECK(abs1(X_exact(i, j) - scale * X(i, j)) <= tol * X_exact(i, j));
    }
}
