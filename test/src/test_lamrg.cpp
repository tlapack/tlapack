/// @file test_lamrg.cpp
/// @author Brian Dang, University of Colorado Denver, USA
/// @brief Test LAMRG.
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
#include <tlapack/lapack/lamrg.hpp>

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

TEMPLATE_TEST_CASE("LU factorization of a general m-by-n matrix, blocked",
                   "[ul_mul]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;  // equivalent to using real_t = real_type<T>;

    // m and n represent no. rows and columns of the matrices we will be testing
    // respectively
    idx_t n = GENERATE(17);
    int sign1 = GENERATE(-1, 1);
    int sign2 = GENERATE(-1, 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis((float)1.0, (float)100.0);
    std::uniform_int_distribution<> neg(0, 1);

    DYNAMIC_SECTION("n = " << n << " sign1 = " << sign1 << " sign2 = " << sign2)
    {
        // Create D Vector
        vector<real_t> d(n);
        // Fill D with random numbers
        for (idx_t i = 0; i < n; i++) {
            d[i] = real_t((neg(gen) == 0 ? -1.0 : 1.0) * dis(gen) / 10.0);
        }

        idx_t mid = n / 2;

        // Create n1 and n2
        vector<real_t> n1(d.begin(), d.begin() + mid);
        vector<real_t> n2(d.begin() + mid, d.end());

        // Sort n1 and n2 sign1 and sign2 indicate how n1 and n2 are sorted
        if (sign1 > 0) {
            sort(n1.begin(), n1.end());
        }
        else {
            sort(n1.begin(), n1.end());
            reverse(n1.begin(), n1.end());
        }

        if (sign2 > 0) {
            sort(n2.begin(), n2.end());
        }
        else {
            sort(n2.begin(), n2.end());
            reverse(n2.begin(), n2.end());
        }

        // Create Vector dlambda
        std::vector<real_t> dlambda;
        dlambda.reserve(n);
        dlambda.insert(dlambda.begin(), n1.begin(), n1.end());
        dlambda.insert(dlambda.begin() + n1.size(), n2.begin(), n2.end());

        // Create INDEXQ
        std::vector<idx_t> indexq(n);
        for (idx_t i = 0; i < n; i++) {
            indexq[i] = i;
        }

        // Merge and Sort for Ascending
        idx_t n1Size = n1.size();
        idx_t n2Size = n2.size();
        lamrg(n1Size, n2Size, dlambda, sign1, sign2, indexq);

        for (idx_t i = 1; i < n; i++) {
            CHECK(dlambda[idx_t(indexq[i - 1])] <= dlambda[idx_t(indexq[i])]);
        }
    }
}
