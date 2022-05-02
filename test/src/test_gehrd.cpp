/// @file test_gehrd.cpp
/// @brief Test hessenberg reduction
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_debugutils.hpp>
#include <tlapack.hpp>
#include <iostream>
#include <iomanip>

using namespace tlapack;

/** Calculates res = Q'*Q - I and the frobenius norm of res
 *
 * @return frobenius norm of res
 *
 * @param[in] Q n by n (almost) orthogonal matrix
 * @param[out] res n by n matrix as defined above
 *
 * @ingroup auxiliary
 */
template <class matrix_t>
real_type<type_t<matrix_t>> check_orthogonality(matrix_t &Q, matrix_t &res)
{
    using T = type_t<matrix_t>;

    // res = I
    tlapack::laset(tlapack::Uplo::General, (T)0.0, (T)1.0, res);
    // res = Q'Q - I
    tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans, (T)1.0, Q, Q, (T)-1.0, res);

    // Compute ||res||_F
    return tlapack::lansy(tlapack::frob_norm, tlapack::Uplo::Upper, res);
}

/** Calculates res = Q'*A*Q - H and the frobenius norm of res relative to the norm of A
 *
 * @return frobenius norm of res
 *
 * @param[in] A n by n matrix
 * @param[in] Q n by n matrix
 * @param[in] H n by n matrix
 * @param[out] res n by n matrix as defined above
 * @param[out] work n by n workspace matrix
 *
 * @ingroup auxiliary
 */
template <class matrix_t>
real_type<type_t<matrix_t>> check_similarity_transform(matrix_t &A, matrix_t &Q, matrix_t &H, matrix_t &res, matrix_t &work)
{
    using T = type_t<matrix_t>;

    // res = Q'*A*Q - H
    tlapack::lacpy(Uplo::General, H, res);
    tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans, (T)1.0, Q, A, (T)0.0, work);
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, (T)1.0, work, Q, (T)-1.0, res);

    // Compute ||res||_F/||A||_F
    return tlapack::lange(tlapack::frob_norm, res) / tlapack::lange(tlapack::frob_norm, A);
}

TEMPLATE_TEST_CASE("Hessenberg reduction is backward stable", "[eigenvalues]", float, double, std::complex<float>, std::complex<double>)
{

    using T = TestType;
    using idx_t = std::size_t;
    using real_t = real_type<T>;
    using tlapack::internal::colmajor_matrix;
    using pair = pair<idx_t, idx_t>;

    // Generate n
    idx_t n = GENERATE(1, 2, 3, 5, 10);
    // Generate ilo and ihi
    idx_t ilo_offset = GENERATE(0, 1);
    idx_t ihi_offset = GENERATE(0, 1);
    idx_t ilo = n > 1 ? ilo_offset : 0;
    idx_t ihi = n > 1 + ilo_offset ? n - ihi_offset : n;

    const real_t eps = uroundoff<real_t>();
    const real_t tol = n * 1.0e2 * eps;

    // Define the matrices and vectors
    std::unique_ptr<T[]> _A(new T[n * n]);
    std::unique_ptr<T[]> _H(new T[n * n]);
    std::unique_ptr<T[]> _Q(new T[n * n]);
    std::unique_ptr<T[]> _res(new T[n * n]);
    std::unique_ptr<T[]> _work(new T[n * n]);

    auto A = colmajor_matrix<T>(&_A[0], n, n);
    auto H = colmajor_matrix<T>(&_H[0], n, n);
    auto Q = colmajor_matrix<T>(&_Q[0], n, n);
    auto res = colmajor_matrix<T>(&_res[0], n, n);
    auto work = colmajor_matrix<T>(&_work[0], n, n);
    std::vector<T> tau(n);
    std::vector<T> workv(n);

    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    // Make sure ilo and ihi correspond to the actual matrix
    for (size_t j = 0; j < ilo; ++j)
        for (size_t i = j + 1; i < n; ++i)
            A(i, j) = (T)0.0;
    for (size_t i = ihi; i < n; ++i)
        for (size_t j = 0; j < i; ++j)
            A(i, j) = (T)0.0;
    tlapack::lacpy(Uplo::General, A, H);

    DYNAMIC_SECTION( "GEHD2 with n = " << n << " ilo = " << ilo << " ihi = " << ihi )
    {
        tlapack::gehd2(ilo, ihi, H, tau, workv);

        // Generate orthogonal matrix Q
        tlapack::lacpy(Uplo::General, H, Q);
        tlapack::unghr(ilo, ihi, Q, tau, workv);

        // Remove junk from lower half of H
        for (size_t j = 0; j < n; ++j)
            for (size_t i = j + 2; i < n; ++i)
                H(i, j) = 0.0;

        // Calculate residuals
        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);

        auto simil_res_norm = check_similarity_transform(A, Q, H, res, work);
        CHECK(simil_res_norm <= tol);
    }
    DYNAMIC_SECTION( "GEHRD with n = " << n << " ilo = " << ilo << " ihi = " << ihi )
    {
        gehrd_opts_t<idx_t, T> opts = {.nb = 2, .nx_switch = 2};
        idx_t required_workspace = get_work_gehrd(ilo, ihi, A, tau, opts);
        std::unique_ptr<T[]> _work2(new T[required_workspace]);
        opts._work = &_work2[0];
        opts.lwork = required_workspace;
        tlapack::gehrd(ilo, ihi, H, tau, opts);

        // Generate orthogonal matrix Q
        tlapack::lacpy(Uplo::General, H, Q);
        tlapack::unghr(ilo, ihi, Q, tau, workv);

        // Remove junk from lower half of H
        for (size_t j = 0; j < n; ++j)
            for (size_t i = j + 2; i < n; ++i)
                H(i, j) = 0.0;

        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);

        auto simil_res_norm = check_similarity_transform(A, Q, H, res, work);
        CHECK(simil_res_norm <= tol);
    }
}