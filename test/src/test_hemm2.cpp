/// @file test_lu_mult.cpp
/// @author Brian Dang, University of Colorado Denver, USA
/// @brief Test LLH multiplication
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "TestUploMatrix.hpp"

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/blas/hemm2.hpp>

using namespace tlapack;

#define TESTUPLO_TYPES_TO_TEST                                          \
    (TestUploMatrix<float, size_t, Uplo::Lower, Layout::ColMajor>),     \
        (TestUploMatrix<float, size_t, Uplo::Upper, Layout::ColMajor>), \
        (TestUploMatrix<float, size_t, Uplo::Lower, Layout::RowMajor>), \
        (TestUploMatrix<float, size_t, Uplo::Upper, Layout::RowMajor>)

TEMPLATE_TEST_CASE("mult a triangular matrix with a rectangular matrix",
                   "[hemm_brian]",
                   TLAPACK_TYPES_TO_TEST,
                   TESTUPLO_TYPES_TO_TEST)

{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;
    using range = pair<idx_t, idx_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    const idx_t n = GENERATE(5, 10, 20, 23, 30);
    const idx_t m = GENERATE(2, 17, 18, 20, 26);

    const Side side = GENERATE(Side::Left, Side::Right);
    const Uplo uplo = GENERATE(Uplo::Lower, Uplo ::Upper);
    const Op op = GENERATE(Op::NoTrans, Op::Trans, Op::ConjTrans);

    DYNAMIC_SECTION("n = " << n << " m = " << m << " side = " << side
                           << " uplo = " << uplo << " op = " << op)
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        // Create Matrices
        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);

        // Update A with random numbers, and make it positive definite
        mm.random(uplo, A);
        for (idx_t j = 0; j < n; ++j)
            A(j, j) += real_t(n);

        mm.random(B);

        // std::vector < T B_;
        // auto B = new_matrix(B_, n, n);
        // std::vector<T> C_;
        // auto C = new_matrix(C_, n, n);

        // if (m <= n) {
        //     const real_t eps = ulp<real_t>();
        //     const real_t tol = real_t(n) * eps;

        //     std::vector<T> C_;
        //     auto C = new_matrix(C_, n, n);
        //     std::vector<T> A_;
        //     auto A = new_matrix(A_, n, n);
        //     std::vector<T> B_;
        //     auto B = new_matrix(B_, n, n);

        //     // Generate n-by-n random matrix
        //     mm.random(A);

        //     lacpy(GENERAL, A, C);
        //     lacpy(GENERAL, A, B);

        //     auto subA = slice(A, range(0, n - 1), range(1, n));
        //     laset(UPPER_TRIANGLE, real_t(0), real_t(0), subA);

        //     real_t normA = lantr(MAX_NORM, LOWER_TRIANGLE, Diag::NonUnit, A);

        //     {
        //         // // A = C *C^H
        //         mult_llh(C);

        //         // C = A*A^H - C
        //         herk(LOWER_TRIANGLE, Op::NoTrans, real_t(1), A, real_t(-1),
        //         C);

        //         // Check if residual is 0 with machine accuracy
        //         real_t llh_mult_res_norm =
        //             lantr(MAX_NORM, LOWER_TRIANGLE, Diag::NonUnit, C);
        //         CHECK(llh_mult_res_norm <= tol * normA);

        //         real_t sum(0);
        //         for (idx_t j = 0; j < n; j++)
        //             for (idx_t i = 0; i < j; i++)
        //                 sum += abs1(B(i, j) - C(i, j));

        //         CHECK(sum == real_t(0));
        //     }
        // }
    }
}
