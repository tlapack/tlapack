/// @file example_interoperability.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// clang-format off
#include <tlapack/plugins/mdspan.hpp>
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/eigen.hpp>
// clang-format on

#include <tlapack/blas/syr.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/getrf.hpp>

// If C++23 is not available, use std::mdspan from kokkos
#if __cplusplus < 202300L
namespace std {
using std::experimental::mdspan;
}
#endif

int main()
{
    using namespace tlapack;

    // Create matrices A and B, and vectors v and piv
    auto A_ = std::vector<double>(100 * 100);
    auto A = std::mdspan(A_.data(), 100, 100);
    auto B = Eigen::MatrixXd::Random(100, 100).eval();
    auto v = std::vector<double>(100);
    auto piv_ = std::vector<int>(100);
    auto piv = LegacyVector<int>(100, piv_.data());

    // Fill A and v with random values
    for (auto& x : A_)
        x = rand() / static_cast<float>(RAND_MAX);
    for (auto& x : v)
        x = rand() / static_cast<float>(RAND_MAX);

    // Mix up values in B
    trmm(LEFT_SIDE, LOWER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, 1.0, A, B);
    syr(UPPER_TRIANGLE, 1.0, v, B);

    // Compute the LU factorization of A and B
    int infoA = getrf(A, piv);
    int infoB = getrf(B, piv);

    return infoA + infoB;
}
