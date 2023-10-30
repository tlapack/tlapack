/// @file test_stevd.cpp Test the symmetric tridiagonal eigenvalue problem solver
/// @author Xuan Jiang, University of California, Berkeley, USA
//
// Copyright 2023, University of California, Berkeley, USA. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/blas/copy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>

// Symmetric tridiagonal eigenvalue problem solver
#include <tlapack/lapack/stevd.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Symmetric Tridiagonal Eigenvalue Problem",
                   "[stevd]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);
    using vector_t = vector_type<TestType>;
    using T = type_t<vector_t>;
    using idx_t = size_type<vector_t>;
    using real_t = real_type<T>;

    // Functor
    Create<vector_t> new_vector;

    // Test parameters
    const idx_t n = GENERATE(10, 20, 50);
    const std::string jobz = GENERATE("N", "V");

    DYNAMIC_SECTION("n = " << n << " jobz = " << jobz)
    {
        // Constants
        const real_t tol = real_t(n) * ulp<real_t>();
        const T zero(0);
        const idx_t ldz = (jobz == "V") ? n : 1;

        // Vectors
        std::vector<T> d_;
        auto d = new_vector(d_, n);
        std::vector<T> e_;
        auto e = new_vector(e_, n - 1);
        std::vector<T> z_;
        auto z = new_vector(z_, n * ldz);
        std::vector<T> work_;
        auto work = new_vector(work_, 1);  // Adjust the size as needed
        std::vector<int> iwork_;
        auto iwork = new_vector(iwork_, 1);  // Adjust the size as needed

        // Initialize d, e, and z
        for (idx_t i = 0; i < n; ++i) {
            d[i] = real_t(rand()) / real_t(RAND_MAX);
            if (i < n - 1) e[i] = real_t(rand()) / real_t(RAND_MAX);
        }
        laset(FULL_MATRIX, n, ldz, zero, zero, z);

        // Solve the symmetric tridiagonal eigenvalue problem
        int info;
        stevd(jobz, n, d, e, z, ldz, work, iwork, info);

        // Check the results
        CHECK(info == 0);

        if (jobz == "V") {
            // Add checks for eigenvalues and eigenvectors here
            // ...

        } else {
            // Add checks for eigenvalues only here
            // ...
        }
    }
}
