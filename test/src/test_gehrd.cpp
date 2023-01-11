/// @file test_gehrd.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test hessenberg reduction
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
#include <tlapack/lapack/gehrd.hpp>
#include <tlapack/lapack/unghr.hpp>

using namespace tlapack;

template <typename matrix_t, typename vector_t>
void check_hess_reduction(size_type<matrix_t> ilo, size_type<matrix_t> ihi, matrix_t H, vector_t tau, matrix_t A)
{
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    idx_t n = ncols(A);

    const real_type<T> eps = uroundoff<real_type<T>>();
    const real_type<T> tol = real_type<T>(n * 1.0e2) * eps;

    std::vector<T> Q_; auto Q = new_matrix( Q_, n, n );
    std::vector<T> res_; auto res = new_matrix( res_, n, n );
    std::vector<T> work_; auto work = new_matrix( work_, n, n );

    // Generate orthogonal matrix Q
    tlapack::lacpy(Uplo::General, H, Q);
    tlapack::unghr(ilo, ihi, Q, tau);

    // Remove junk from lower half of H
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 2; i < n; ++i)
            H(i, j) = T(0);

    // Calculate residuals
    auto orth_res_norm = check_orthogonality(Q, res);
    CHECK(orth_res_norm <= tol);

    auto normA = tlapack::lange(tlapack::frob_norm, A);
    auto simil_res_norm = check_similarity_transform(A, Q, H, res, work);
    CHECK(simil_res_norm <= tol * normA);
}

TEMPLATE_TEST_CASE("Hessenberg reduction is backward stable", "[eigenvalues][hessenberg]", TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    auto matrix_type = GENERATE(as<std::string>{}, "Random", "Near overflow");

    rand_generator gen;
    idx_t n = 0;
    idx_t ilo = 0;
    idx_t ihi = 0;
    if (matrix_type == "Random")
    {
        // Generate n
        n = GENERATE(1, 2, 3, 5, 10);
        // Generate ilo and ihi
        idx_t ilo_offset = GENERATE(0, 1);
        idx_t ihi_offset = GENERATE(0, 1);
        ilo = n > 1 ? ilo_offset : 0;
        ihi = n > 1 + ilo_offset ? n - ihi_offset : n;
    }
    if (matrix_type == "Near overflow")
    {
        n = 5;
        ilo = 0;
        ihi = n;
    }

    // Define the matrices and vectors
    std::vector<T> A_; auto A = new_matrix( A_, n, n );
    std::vector<T> H_; auto H = new_matrix( H_, n, n );
    std::vector<T> tau(n);

    if (matrix_type == "Random")
    {
        // Generate a random matrix in A
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                A(i, j) = rand_helper<T>(gen);
    }
    if (matrix_type == "Near overflow")
    {
        const real_t large_num = safe_max<real_t>() * uroundoff<real_t>();

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                A(i, j) = large_num;
    }

    // Make sure ilo and ihi correspond to the actual matrix
    for (idx_t j = 0; j < ilo; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = (T)0.0;
    for (idx_t i = ihi; i < n; ++i)
        for (idx_t j = 0; j < i; ++j)
            A(i, j) = (T)0.0;
    tlapack::lacpy(Uplo::General, A, H);

    INFO("matrix = " << matrix_type << " n = " << n << " ilo = " << ilo << " ihi = " << ihi);
    
    SECTION("GEHD2")
    {    
        tlapack::gehd2(ilo, ihi, H, tau);

        check_hess_reduction(ilo, ihi, H, tau, A);
    }
    
    SECTION("GEHRD")
    {
        idx_t nb = GENERATE(2, 3);
        INFO("nb = " << nb);
        
        gehrd_opts_t<idx_t> opts;
        opts.nb = nb;
        opts.nx_switch = 2;
        tlapack::gehrd(ilo, ihi, H, tau, opts);

        check_hess_reduction(ilo, ihi, H, tau, A);
    }
}
