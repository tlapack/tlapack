/// @file test_laed4.cpp
/// @author Brian Dang, University of Colorado Denver, USA
/// @brief Test LAED4.
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <algorithm>

// Other routines
#include <tlapack/lapack/laed2.hpp>
#include <tlapack/lapack/laset.hpp>

using namespace tlapack;
using namespace std;

template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < n; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

TEMPLATE_TEST_CASE(
    "LAED4",
    "[stedc,laed4]",
    (tlapack::LegacyMatrix<double, std::size_t, tlapack::Layout::ColMajor>))
// TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;  // equivalent to using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    // m and n represent no. rows and columns of the matrices we will be testing
    // respectively
    // idx_t n = GENERATE(2, 5, 30, 50, 100);
    idx_t n = GENERATE(7);

    srand(3);
    // real_t rho = real_t(GENERATE(15.7, 100));
    real_t rho = real_t(GENERATE(15.7));
    DYNAMIC_SECTION("n = " << n << " rho = " << rho)
    {
        // eps is the machine precision, and tol is the tolerance we accept
        // for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(20 * n) * eps;

        const idx_t n1 = 3;
        const idx_t ldq = n;

        idx_t k = 0;

        std::vector<real_t> d(n);
        std::vector<real_t> Q_(n);
        auto Q = new_matrix(Q_, n, n);
        laset(GENERAL, real_t(0), real_t(1), Q);

        // std::vector<real_t> Q2(n1 * n1 + (n - n1) * (n - n1));
        std::vector<real_t> Q2(n * n);

        std::vector<real_t> z(n);
        std::vector<real_t> w(n);
        std::vector<idx_t> indxq(n), indx(n), indxc(n), indxp(n), coltyp(n);
        for (idx_t i = 0; i < n; i++) {
            indx[i] = i;
            indxq[i] = i;
            indxc[i] = i;
            indxp[i] = i;
            w[i] = real_t(0);
            d[i] = i + 1;
            z[i] = real_t(0.5 + i);
        }

        std::vector<real_t> dlambda(n);

        int info = tlapack::laed2(k, n, n1, d, Q, ldq, indxq, rho, z, dlambda,
                                  w, Q2, indx, indxc, indxp, coltyp);
    }
}
