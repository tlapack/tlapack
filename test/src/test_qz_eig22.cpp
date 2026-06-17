/// @file test_qz_eig22.cpp Test the solution of 2x2 generalized eigenvalue
/// problems
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/blas/rot.hpp>
#include <tlapack/blas/rotg.hpp>
#include <tlapack/lapack/lahqz_eig22.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE(
    "check that lahqz_eig22 gives correct eigenvalues on random 2x2 "
    "pencils",
    "[generalizedeigenvalues]",
    TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using complex_t = complex_type<real_t>;

    // Functor
    Create<matrix_t> new_matrix;
    Create<complex_type<matrix_t>> new_complex_matrix;

    // MatrixMarket reader
    uint64_t seed = GENERATE(1, 2, 3, 4, 5, 6);
    bool complex_conjugate = GENERATE(true, false);
    bool singular0 = GENERATE(true, false);
    bool singular1 = GENERATE(true, false);
    if constexpr (is_complex<T>) {
        complex_conjugate = false;
    }

    DYNAMIC_SECTION(" seed = " << seed << " complex conjugate ? "
                               << complex_conjugate << " singular0 ? "
                               << singular0 << " singular1 ? " << singular1)
    {
        MatrixMarket mm;
        mm.gen.seed(seed);

        const real_type<T> eps = ulp<real_type<T>>();

        std::vector<T> A_;
        auto A = new_matrix(A_, 2, 2);
        std::vector<T> B_;
        auto B = new_matrix(B_, 2, 2);

        // Generate pencil with prescibed eigenvalues

        real_t beta1_e = rand_helper<real_t>(mm.gen);
        real_t beta2_e = rand_helper<real_t>(mm.gen);

        if (singular0) {
            beta1_e = real_t(0);
        }
        if (singular1) {
            beta2_e = real_t(0);
        }

        complex_t alpha1_e, alpha2_e;
        if constexpr (is_complex<T>) {
            alpha1_e = complex_t(rand_helper<real_t>(mm.gen),
                                 rand_helper<real_t>(mm.gen));
            alpha2_e = complex_t(rand_helper<real_t>(mm.gen),
                                 rand_helper<real_t>(mm.gen));

            A(0, 0) = alpha1_e;
            A(1, 0) = T(0);
            A(0, 1) = rand_helper<T>(mm.gen);
            A(1, 1) = alpha2_e;

            B(0, 0) = beta1_e;
            B(1, 1) = beta2_e;
            B(0, 1) = rand_helper<T>(mm.gen);
            B(1, 0) = T(0);
        }
        else {
            if (complex_conjugate) {
                real_t alpha1_real = rand_helper<real_t>(mm.gen);
                real_t alpha1_imag = rand_helper<real_t>(mm.gen);
                alpha1_e = complex_t(alpha1_real, alpha1_imag);
                alpha2_e = complex_t(alpha1_real, -alpha1_imag);

                beta2_e = beta1_e;

                A(0, 0) = alpha1_real;
                A(1, 1) = alpha1_real;

                real_t temp = rand_helper<real_t>(mm.gen);
                A(0, 1) = alpha1_imag * alpha1_imag / temp;
                A(1, 0) = -temp;

                B(0, 0) = beta1_e;
                B(1, 1) = beta2_e;
                B(0, 1) = T(0);
                B(1, 0) = T(0);
            }
            else {
                alpha1_e = rand_helper<real_t>(mm.gen);
                alpha2_e = rand_helper<real_t>(mm.gen);

                A(0, 0) = real(alpha1_e);
                A(1, 0) = T(0);
                A(0, 1) = rand_helper<T>(mm.gen);
                A(1, 1) = real(alpha2_e);

                B(0, 0) = beta1_e;
                B(1, 1) = beta2_e;
                B(0, 1) = rand_helper<T>(mm.gen);
                B(1, 0) = T(0);
            }
        }

        // Apply random orthogonal transformation (rotations) to the pencil to
        // make it more general
        {
            T x = rand_helper<T>(mm.gen);
            T y = rand_helper<T>(mm.gen);
            real_t c;
            T s;
            rotg(x, y, c, s);

            auto a1 = col(A, 0);
            auto a2 = col(A, 1);
            rot(a1, a2, c, s);

            auto b1 = col(B, 0);
            auto b2 = col(B, 1);
            rot(b1, b2, c, s);
        }
        {
            T x = B(0, 0);
            T y = B(1, 0);
            real_t c;
            T s;
            rotg(x, y, c, s);

            auto a1 = row(A, 0);
            auto a2 = row(A, 1);
            rot(a1, a2, c, s);

            auto b1 = row(B, 0);
            auto b2 = row(B, 1);
            rot(b1, b2, c, s);

            B(1, 0) = T(0);
        }

        real_t beta1, beta2;
        std::complex<real_t> alpha1, alpha2;
        lahqz_eig22(A, B, alpha1, alpha2, beta1, beta2);

        // Check that the computed eigenvalues are close to the expected
        // eigenvalues
        // We need to take into account that the eigenvalues may not be in the
        // same order, so we check both possibilities after normalizing

        auto [err1, err2] = check_generalized_eigenvalues(
            alpha1, alpha2, beta1, beta2, alpha1_e, alpha2_e, beta1_e, beta2_e);

        CHECK(err1 < 10 * eps);
        CHECK(err2 < 10 * eps);
    }
}