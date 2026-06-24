/// @file test_qz_schur22.cpp Test the solution of 2x2 generalized eigenvalue
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
#include <tlapack/lapack/lahqz_schur22.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE(
    "check that lahqz_schur22 gives correct eigenvalues on random 2x2 "
    "pencils with eigenvalues that can be deflated",
    "[generalizedeigenvalues]",
    TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<TA>;
    using complex_t = complex_type<real_t>;

    // Functor
    Create<matrix_t> new_matrix;
    Create<complex_type<matrix_t>> new_complex_matrix;

    // MatrixMarket reader
    uint64_t seed = GENERATE(1, 2, 3, 4, 5, 6);
    auto matrix_type = GENERATE("Adefl", "Infinite0", "Infinite1");

    DYNAMIC_SECTION(" seed = " << seed << " matrix_type = " << matrix_type)
    {
        MatrixMarket mm;
        mm.gen.seed(seed);

        const real_type<TA> eps = ulp<real_type<TA>>();

        std::vector<TA> A_;
        auto A = new_matrix(A_, 2, 2);
        std::vector<TA> B_;
        auto B = new_matrix(B_, 2, 2);

        mm.random(A);
        mm.random(B);

        B(1, 0) = TA(0);

        if (matrix_type == "Adefl") {
            A(1, 0) = TA(0.01) * eps;
        }
        else if (matrix_type == "Infinite0") {
            B(0, 0) = TA(0.01) * eps;
        }
        else if (matrix_type == "Infinite1") {
            B(1, 1) = TA(0.01) * eps;
        }

        std::vector<TA> S_;
        auto S = new_matrix(S_, 2, 2);
        std::vector<TA> T_;
        auto T = new_matrix(T_, 2, 2);
        std::vector<TA> Q_;
        auto Q = new_matrix(Q_, 2, 2);
        std::vector<TA> Z_;
        auto Z = new_matrix(Z_, 2, 2);

        lacpy(GENERAL, A, S);
        lacpy(GENERAL, B, T);

        real_t beta1, beta2;
        complex_t alpha1, alpha2;

        real_t cl, cr;
        TA sl, sr, scal0, scal1;

        lahqz_schur22(S, T, alpha1, alpha2, beta1, beta2, cl, sl, cr, sr, scal0,
                      scal1);

        Q(0, 0) = cl;
        Q(0, 1) = -sl;
        Q(1, 0) = conj(sl);
        Q(1, 1) = cl;

        Z(0, 0) = cr * scal0;
        Z(0, 1) = -conj(sr) * scal1;
        Z(1, 0) = sr * scal0;
        Z(1, 1) = cr * scal1;

        // Check that the eigenvalues match with the diagonal elements
        CHECK(abs1(alpha1 - S(0, 0)) <= eps * max(real_t(1), abs1(S(0, 0))));
        CHECK(abs1(beta1 - T(0, 0)) <= eps * max(real_t(1), abs1(T(0, 0))));
        CHECK(abs1(alpha2 - S(1, 1)) <= eps * max(real_t(1), abs1(S(1, 1))));
        CHECK(abs1(beta2 - T(1, 1)) <= eps * max(real_t(1), abs1(T(1, 1))));

        // Check that the diagonal of B is real and non-negative
        CHECK(real(T(0, 0)) >= real_t(0));
        CHECK(real(T(1, 1)) >= real_t(0));
        CHECK(imag(T(0, 0)) == real_t(0));
        CHECK(imag(T(1, 1)) == real_t(0));

        // Check backward error of the generalized eigenvalue problem
        std::vector<TA> res_;
        auto res = new_matrix(res_, 2, 2);
        std::vector<TA> work_;
        auto work = new_matrix(work_, 2, 2);
        auto errA =
            check_generalized_similarity_transform(A, Q, Z, S, res, work);
        auto errB =
            check_generalized_similarity_transform(B, Q, Z, T, res, work);

        auto normA = tlapack::lange(tlapack::FROB_NORM, A);
        auto normB = tlapack::lange(tlapack::FROB_NORM, B);

        CHECK(errA <= real_t(10) * eps * normA);
        CHECK(errB <= real_t(10) * eps * normB);
    }
}

TEMPLATE_TEST_CASE(
    "check that lahqz_schur22 gives correct eigenvalues on random 2x2 "
    "pencils with eigenvalues that cannot immediately be deflated",
    "[generalizedeigenvalues]",
    TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<TA>;
    using complex_t = complex_type<real_t>;

    // Functor
    Create<matrix_t> new_matrix;
    Create<complex_type<matrix_t>> new_complex_matrix;

    // MatrixMarket reader
    uint64_t seed = GENERATE(1, 2, 3, 4, 5, 6);

    DYNAMIC_SECTION(" seed = " << seed)
    {
        MatrixMarket mm;
        mm.gen.seed(seed);

        const real_type<TA> eps = ulp<real_type<TA>>();

        std::vector<TA> A_;
        auto A = new_matrix(A_, 2, 2);
        std::vector<TA> B_;
        auto B = new_matrix(B_, 2, 2);

        real_t beta1_ref = rand_helper<real_t>(mm.gen);
        real_t beta2_ref = rand_helper<real_t>(mm.gen);
        // Note the use of TA here
        TA alpha1_ref = rand_helper<TA>(mm.gen);
        TA alpha2_ref = rand_helper<TA>(mm.gen);

        A(0, 0) = alpha1_ref;
        A(1, 1) = alpha2_ref;
        A(0, 1) = rand_helper<TA>(mm.gen);
        A(1, 0) = TA(0);

        B(0, 0) = beta1_ref;
        B(1, 1) = beta2_ref;
        B(0, 1) = rand_helper<TA>(mm.gen);
        B(1, 0) = TA(0);

        // Apply random orthogonal transformation (rotations) to the pencil to
        // make it more general
        {
            TA x = rand_helper<TA>(mm.gen);
            TA y = rand_helper<TA>(mm.gen);
            real_t c;
            TA s;
            rotg(x, y, c, s);

            auto a1 = col(A, 0);
            auto a2 = col(A, 1);
            rot(a1, a2, c, s);

            auto b1 = col(B, 0);
            auto b2 = col(B, 1);
            rot(b1, b2, c, s);
        }
        {
            TA x = B(0, 0);
            TA y = B(1, 0);
            real_t c;
            TA s;
            rotg(x, y, c, s);

            auto a1 = row(A, 0);
            auto a2 = row(A, 1);
            rot(a1, a2, c, s);

            auto b1 = row(B, 0);
            auto b2 = row(B, 1);
            rot(b1, b2, c, s);

            B(1, 0) = TA(0);
        }

        std::vector<TA> S_;
        auto S = new_matrix(S_, 2, 2);
        std::vector<TA> T_;
        auto T = new_matrix(T_, 2, 2);
        std::vector<TA> Q_;
        auto Q = new_matrix(Q_, 2, 2);
        std::vector<TA> Z_;
        auto Z = new_matrix(Z_, 2, 2);

        lacpy(GENERAL, A, S);
        lacpy(GENERAL, B, T);

        real_t beta1, beta2;
        complex_t alpha1, alpha2;

        real_t cl, cr;
        TA sl, sr, scal0, scal1;

        lahqz_schur22(S, T, alpha1, alpha2, beta1, beta2, cl, sl, cr, sr, scal0,
                      scal1);

        Q(0, 0) = cl;
        Q(0, 1) = -sl;
        Q(1, 0) = conj(sl);
        Q(1, 1) = cl;

        Z(0, 0) = cr * scal0;
        Z(0, 1) = -conj(sr) * scal1;
        Z(1, 0) = sr * scal0;
        Z(1, 1) = cr * scal1;

        // Check that the eigenvalues match with the diagonal elements
        CHECK(abs1(alpha1 - S(0, 0)) <= eps * max(real_t(1), abs1(S(0, 0))));
        CHECK(abs1(beta1 - T(0, 0)) <= eps * max(real_t(1), abs1(T(0, 0))));
        CHECK(abs1(alpha2 - S(1, 1)) <= eps * max(real_t(1), abs1(S(1, 1))));
        CHECK(abs1(beta2 - T(1, 1)) <= eps * max(real_t(1), abs1(T(1, 1))));

        // Check that the diagonal of B is real and non-negative
        CHECK(real(T(0, 0)) >= real_t(0));
        CHECK(real(T(1, 1)) >= real_t(0));
        CHECK(imag(T(0, 0)) == real_t(0));
        CHECK(imag(T(1, 1)) == real_t(0));

        // Check backward error of the generalized eigenvalue problem
        std::vector<TA> res_;
        auto res = new_matrix(res_, 2, 2);
        std::vector<TA> work_;
        auto work = new_matrix(work_, 2, 2);
        auto errA =
            check_generalized_similarity_transform(A, Q, Z, S, res, work);
        auto errB =
            check_generalized_similarity_transform(B, Q, Z, T, res, work);

        auto normA = tlapack::lange(tlapack::FROB_NORM, A);
        auto normB = tlapack::lange(tlapack::FROB_NORM, B);

        CHECK(errA <= real_t(10) * eps * normA);
        CHECK(errB <= real_t(10) * eps * normB);

        // Check that the computed eigenvalues are close to the expected
        // eigenvalues
        complex_t alpha1_ref_c = alpha1_ref;
        complex_t alpha2_ref_c = alpha2_ref;

        auto [err1, err2] = check_generalized_eigenvalues(
            alpha1, alpha2, beta1, beta2, alpha1_ref_c, alpha2_ref_c, beta1_ref,
            beta2_ref);

        CHECK(err1 < 10 * eps);
        CHECK(err2 < 10 * eps);
    }
}

TEMPLATE_TEST_CASE(
    "check that lahqz_schur22 gives correct eigenvalues on random 2x2 "
    "real pencils with complex eigenvalues",
    "[generalizedeigenvalues]",
    TLAPACK_REAL_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<TA>;
    using complex_t = complex_type<real_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    uint64_t seed = GENERATE(1, 2, 3, 4, 5, 6);

    DYNAMIC_SECTION(" seed = " << seed)
    {
        MatrixMarket mm;
        mm.gen.seed(seed);

        const real_type<TA> eps = ulp<real_type<TA>>();

        std::vector<TA> A_;
        auto A = new_matrix(A_, 2, 2);
        std::vector<TA> B_;
        auto B = new_matrix(B_, 2, 2);

        real_t beta1_ref = real_t(1);
        real_t beta2_ref = real_t(1);
        // Note the use of TA here
        real_t alpha_r = rand_helper<real_t>(mm.gen);
        real_t alpha_i = rand_helper<real_t>(mm.gen);

        complex_t alpha1_ref(alpha_r, alpha_i);
        complex_t alpha2_ref(alpha_r, -alpha_i);

        A(0, 0) = alpha_r;
        A(1, 1) = alpha_r;
        A(0, 1) = rand_helper<TA>(mm.gen);
        A(1, 0) = -alpha_i * alpha_i / A(0, 1);

        B(0, 0) = beta1_ref;
        B(1, 1) = beta2_ref;
        B(0, 1) = TA(0);
        B(1, 0) = TA(0);

        // Apply random orthogonal transformation (rotations) to the pencil to
        // make it more general
        {
            TA x = rand_helper<TA>(mm.gen);
            TA y = rand_helper<TA>(mm.gen);
            real_t c;
            TA s;
            rotg(x, y, c, s);

            auto a1 = col(A, 0);
            auto a2 = col(A, 1);
            rot(a1, a2, c, s);

            auto b1 = col(B, 0);
            auto b2 = col(B, 1);
            rot(b1, b2, c, s);
        }
        {
            TA x = B(0, 0);
            TA y = B(1, 0);
            real_t c;
            TA s;
            rotg(x, y, c, s);

            auto a1 = row(A, 0);
            auto a2 = row(A, 1);
            rot(a1, a2, c, s);

            auto b1 = row(B, 0);
            auto b2 = row(B, 1);
            rot(b1, b2, c, s);

            B(1, 0) = TA(0);
        }

        std::vector<TA> S_;
        auto S = new_matrix(S_, 2, 2);
        std::vector<TA> T_;
        auto T = new_matrix(T_, 2, 2);
        std::vector<TA> Q_;
        auto Q = new_matrix(Q_, 2, 2);
        std::vector<TA> Z_;
        auto Z = new_matrix(Z_, 2, 2);

        lacpy(GENERAL, A, S);
        lacpy(GENERAL, B, T);

        real_t beta1, beta2;
        complex_t alpha1, alpha2;

        real_t cl, cr;
        TA sl, sr, scal0, scal1;

        lahqz_schur22(S, T, alpha1, alpha2, beta1, beta2, cl, sl, cr, sr, scal0,
                      scal1);

        Q(0, 0) = cl;
        Q(0, 1) = -sl;
        Q(1, 0) = conj(sl);
        Q(1, 1) = cl;

        Z(0, 0) = cr * scal0;
        Z(0, 1) = -conj(sr) * scal1;
        Z(1, 0) = sr * scal0;
        Z(1, 1) = cr * scal1;

        // Check that the diagonal of B is real and non-negative
        CHECK(real(T(0, 0)) >= real_t(0));
        CHECK(real(T(1, 1)) >= real_t(0));
        CHECK(imag(T(0, 0)) == real_t(0));
        CHECK(imag(T(1, 1)) == real_t(0));

        // Check backward error of the generalized eigenvalue problem
        std::vector<TA> res_;
        auto res = new_matrix(res_, 2, 2);
        std::vector<TA> work_;
        auto work = new_matrix(work_, 2, 2);
        auto errA =
            check_generalized_similarity_transform(A, Q, Z, S, res, work);
        auto errB =
            check_generalized_similarity_transform(B, Q, Z, T, res, work);

        auto normA = tlapack::lange(tlapack::FROB_NORM, A);
        auto normB = tlapack::lange(tlapack::FROB_NORM, B);

        CHECK(errA <= real_t(10) * eps * normA);
        CHECK(errB <= real_t(10) * eps * normB);

        // Check that the computed eigenvalues are close to the expected
        // eigenvalues
        auto [err1, err2] = check_generalized_eigenvalues(
            alpha1, alpha2, beta1, beta2, alpha1_ref, alpha2_ref, beta1_ref,
            beta2_ref);

        CHECK(err1 < 10 * eps);
        CHECK(err2 < 10 * eps);
    }
}