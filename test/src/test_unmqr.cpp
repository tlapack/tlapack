/// @file test_unmqr.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test unmqr
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
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/ungqr.hpp>
#include <tlapack/lapack/unmqr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Multiply m-by-n matrix with orthogonal QR factor",
                   "[unmqr]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    idx_t m = GENERATE(5, 10);
    idx_t n = GENERATE(1, 5, 10);
    idx_t k = min(m, n);
    idx_t k2 = GENERATE(1, 4, 5, 10);
    idx_t nb = GENERATE(1, 2, 3);

    Side side = GENERATE(Side::Left, Side::Right);
    Op trans = GENERATE(Op::NoTrans, Op::ConjTrans);

    idx_t mc, nc;
    if (side == Side::Left) {
        mc = m;
        nc = k2;
    }
    else {
        mc = k2;
        nc = m;
    }

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(100.0 * max(mc, nc)) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> C_;
    auto C = new_matrix(C_, mc, nc);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, m);

    std::vector<T> tau(k);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    for (idx_t j = 0; j < nc; ++j)
        for (idx_t i = 0; i < mc; ++i)
            C(i, j) = rand_helper<T>();

    DYNAMIC_SECTION("m = " << m << " n = " << n << " side = " << side
                           << " trans = " << trans << " k2 = " << k2
                           << " nb = " << nb)
    {
        // QR factorization
        geqr2(A, tau);

        // Calculate the result of unmqr using ung2r and gemm
        for (idx_t j = 0; j < k; ++j)
            for (idx_t i = 0; i < m; ++i)
                Q(i, j) = A(i, j);
        ungqr_opts_t<> ungqrOpts;
        ungqrOpts.nb = nb;
        ungqr(Q, tau, ungqrOpts);

        std::vector<T> Cq_;
        auto Cq = new_matrix(Cq_, mc, nc);
        laset(Uplo::General, T(0.), T(0.), Cq);
        if (side == Side::Left)
            gemm(trans, Op::NoTrans, T(1.), Q, C, T(0.), Cq);
        else
            gemm(Op::NoTrans, trans, T(1.), C, Q, T(0.), Cq);

        // Run the routine we are testing
        unmqr_opts_t<> unmqrOpts;
        unmqrOpts.nb = nb;
        unmqr(side, trans, cols(A, range(0, k)), tau, C, unmqrOpts);

        // Compare results
        for (idx_t j = 0; j < nc; ++j)
            for (idx_t i = 0; i < mc; ++i)
                C(i, j) -= Cq(i, j);
        real_t repres = lange(Norm::Max, C);
        CHECK(repres <= tol);
    }
}
