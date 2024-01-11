/// @file test_hetd2.cpp
/// @author Skylar Johns, University of Colorado Denver, USA
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @brief Test HETD2
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
// #include <tlapack/plugins/debugutils.hpp>

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/hetd2.hpp>
#include <tlapack/lapack/ungtr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Tridiagnolization of a symmetric matrix works",
                   "[hetd2]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    // Generators
    idx_t n = GENERATE(6, 13, 29);
    const Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);

    DYNAMIC_SECTION("n = " << n << " uplo = " << uplo)
    {
        // Constants
        const real_t zero(0);
        const real_t one(1);
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(2 * n) * eps;

        // Matrices and vectors
        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);
        std::vector<T> Q_;
        auto Q = new_matrix(Q_, n, n);
        std::vector<real_t> E(n - 1), D(n);
        std::vector<T> tau(n - 1);

        // Fill A with random values
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                A(i, j) = rand_helper<T>();

        // Compute the norm of A
        real_t normA = lange(Norm::Fro, A);

        // Copy A to Q and run the algorithm in Q
        lacpy(uplo, A, Q);
        hetd2(uplo, Q, tau);

        // Store D and test that the diagonal of Q is real
        bool main_diag_is_real = true;
        for (idx_t i = 0; i < n; ++i) {
            const T& Qii = Q(i, i);
            main_diag_is_real =
                main_diag_is_real && (tlapack::abs(imag(Qii)) == zero);

            D[i] = real(Qii);
        }
        REQUIRE(main_diag_is_real);

        // Store E and test that the off-diagonal of Q is real
        bool off_diag_is_real = true;
        for (idx_t i = 0; i < n - 1; ++i) {
            const T& Qij = (uplo == Uplo::Lower) ? Q(i + 1, i) : Q(i, i + 1);
            off_diag_is_real = off_diag_is_real && (tlapack::abs(imag(Qij)) <
                                                    tol * tlapack::abs(Qij));

            E[i] = real(Qij);
        }
        REQUIRE(off_diag_is_real);

        // for (idx_t i = 0; i < n; ++i) {
        //     std::cout << std::endl;
        //     for (idx_t j = 0; j < n; ++j)
        //         std::cout << std::setw(3) << Q(i, j) << " ";
        //     // std::printf("%+8.4f",Q(i,j));
        // }
        // std::cout << std::endl;

        // Compute Q and check that it is orthogonal
        ungtr(uplo, Q, tau);
        auto orth_Q = check_orthogonality(Q);
        CHECK(orth_Q <= tol);

        // Compute A - QBQ^H
        {
            // Auxiliary matrix
            std::vector<T> R_;
            auto R = new_matrix(R_, n, n);

            // Compute R = QB
            for (idx_t i = 0; i < n; ++i) {
                R(i, 0) = Q(i, 0) * D[0] + Q(i, 1) * E[0];
                for (idx_t j = 1; j < n - 1; ++j) {
                    R(i, j) = Q(i, j - 1) * E[j - 1] + Q(i, j) * D[j] +
                              Q(i, j + 1) * E[j];
                }
                R(i, n - 1) = Q(i, n - 2) * E[n - 2] + Q(i, n - 1) * D[n - 1];
            }

            // Make A hermitian
            if (uplo == Uplo::Upper) {
                for (idx_t i = 0; i < n; ++i) {
                    for (idx_t j = 0; j < i; ++j)
                        A(i, j) = conj(A(j, i));
                    A(i, i) = real(A(i, i));
                }
            }
            else {
                for (idx_t i = 0; i < n; ++i) {
                    for (idx_t j = i + 1; j < n; ++j)
                        A(i, j) = conj(A(j, i));
                    A(i, i) = real(A(i, i));
                }
            }

            // Compute A - QBQ^H
            gemm(NO_TRANS, CONJ_TRANS, one, R, Q, -one, A);

            // Check that the error is close to zero
            CHECK(lange(Norm::Fro, A) / normA < tol);
        }
    }
}