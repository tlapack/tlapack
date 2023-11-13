/// @file test_cauchy.cpp
/// @author Aslak Djupskås
/// @brief Test Cauchy invertion 
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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

TEMPLATE_TEST_CASE("Cauchy matrix properties",
                   "[getri][cauchy]", 
                   TLAPACK_TYPES_TO_TEST) 
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;  // equivalent to using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    idx_t n = GENERATE(3, 9, 16);
    DYNAMIC_SECTION("n = " << n) 
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n * n) * eps;

        // Define the matrices and vectors
        std::vector<T> C_;
        auto C = new_matrix(C_, n, n);
        std::vector<T> invCexpl_;
        auto invCexpl = new_matrix(invCexpl_, n, n);
        std::vector<T> x(n);
        std::vector<T> y(n);



        for (idx_t i = 0; i < n; ++i)
        {
            x[i] = (T)(i+1);
            y[i] = (T)((i+1+n));
        }

        // Initialize and compute C using the explicit formula
        mm.generateCauchy(C, x, y); 
        mm.generateInverseCauchy(invCexpl, x, y);
        // make a deep copy C


        // calculate norm of C for later use in relative error
        real_t normC = tlapack::lange(tlapack::MAX_NORM, C);

        // building error matrix E 
        std::vector<T> E_;
        auto E = new_matrix(E_, n, n);

        // E <----- inv(C)*C - I
        gemm(NO_TRANS, NO_TRANS, real_t(1), C, invCexpl, E);
        for (idx_t i = 0; i < n; i++)
            E(i, i) -= real_t(1);


        // error is  || inv(C)*C - I || / ( ||C|| * ||inv(C)|| )
        real_t error = tlapack::lange(tlapack::MAX_NORM, E) /
                       (normC * tlapack::lange(tlapack::MAX_NORM, invCexpl));

        UNSCOPED_INFO("|| inv(C)*C - I || / ( ||C|| * ||inv(C)|| )");
        CHECK(error / tol <= real_t(1));  // tests if error<=tol

        // E <----- C*inv(C) - I
        gemm(NO_TRANS, NO_TRANS, real_t(1), invCexpl, C, E);
        for (idx_t i = 0; i < n; i++)
            E(i, i) -= real_t(1);

        // error is  || C*inv(C) - I || / ( ||C|| * ||inv(C)|| )
        error = tlapack::lange(tlapack::MAX_NORM, E) /
                (normC * tlapack::lange(tlapack::MAX_NORM, invCexpl));

        UNSCOPED_INFO("|| C*inv(C) - I || / ( ||C|| * ||inv(C)|| )");
        CHECK(error / tol <= real_t(1));  // tests if error<=tol
    }
}

