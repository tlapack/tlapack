/// @file test_getri.cpp
/// @author Ali Lotfi, University of Colorado Denver, USA
/// @brief Test functions that calculate inverse of matrices such as getri
/// family.
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
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/getri.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Inversion of a general m-by-n matrix",
                   "[getri]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;  // equivalent to using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // n represent no. rows and columns of the square matrices we will
    // performing tests on
    idx_t n = GENERATE(5, 10, 20, 100);
    GetriVariant variant = GENERATE(GetriVariant::UXLI, GetriVariant::UILI);

    DYNAMIC_SECTION("n = " << n << " variant = " << (char)variant)
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n * n) * eps;

        // Initialize matrices A, and invA to run tests on
        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);
        std::vector<T> invA_;
        auto invA = new_matrix(invA_, n, n);

        // forming A, a random matrix
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i) {
                A(i, j) = rand_helper<T>();
            }

        // make a deep copy A
        lacpy(Uplo::General, A, invA);

        // calculate norm of A for later use in relative error
        real_t norma = tlapack::lange(tlapack::Norm::Max, A);

        // LU factorize Pivoted A
        std::vector<idx_t> piv(n, idx_t(0));
        getrf(invA, piv);

        // run inverse function, this could test any inverse function of choice
        getri_opts_t opts;
        opts.variant = variant;
        getri(invA, piv, opts);

        // building error matrix E
        std::vector<T> E_;
        auto E = new_matrix(E_, n, n);

        // E <----- inv(A)*A - I
        gemm(Op::NoTrans, Op::NoTrans, real_t(1), A, invA, E);
        for (idx_t i = 0; i < n; i++)
            E(i, i) -= real_t(1);

        // error is  || inv(A)*A - I || / ( ||A|| * ||inv(A)|| )
        real_t error = tlapack::lange(tlapack::Norm::Max, E) /
                       (norma * tlapack::lange(tlapack::Norm::Max, invA));

        UNSCOPED_INFO("|| inv(A)*A - I || / ( ||A|| * ||inv(A)|| )");
        CHECK(error / tol <= real_t(1));  // tests if error<=tol

        // E <----- A*inv(A) - I
        gemm(Op::NoTrans, Op::NoTrans, real_t(1), invA, A, E);
        for (idx_t i = 0; i < n; i++)
            E(i, i) -= real_t(1);

        // error is  || A*inv(A) - I || / ( ||A|| * ||inv(A)|| )
        error = tlapack::lange(tlapack::Norm::Max, E) /
                (norma * tlapack::lange(tlapack::Norm::Max, invA));

        UNSCOPED_INFO("|| A*inv(A) - I || / ( ||A|| * ||inv(A)|| )");
        CHECK(error / tol <= real_t(1));  // tests if error<=tol
    }
}
