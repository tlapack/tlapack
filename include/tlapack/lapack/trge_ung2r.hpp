/// @file trge_ung2r.hpp
/// @author Julien Langou, L. Carlos Gutierrez, University of Colorado Denver,
/// USA
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TRGE_UNG2R_HH
#define TLAPACK_TRGE_UNG2R_HH

#include "tlapack/base/utils.hpp"

using namespace tlapack;

template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
void trge_ung2r(matrix_t& W, vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    idx_t n = tau.size();
    idx_t m = nrows(W) - n;
    idx_t k = ncols(W);

    auto Winit0 = slice(W, range{0, n}, range{n, k});
    laset(GENERAL, real_t(0.0), real_t(0.0), Winit0);

    auto Winit1 = slice(W, range{n, m + n}, range{n, k});
    laset(GENERAL, real_t(0.0), real_t(1.0), Winit1);

    auto v = slice(W, range{n, m + n}, n - 1);
    auto W0 = slice(W, n - 1, range{n, k});
    auto W1 = slice(W, range{n, m + n}, range{n, k});

    std::vector<T> work_;
    auto work = new_matrix(work_, m + n, 1);

    if (k > n)
        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v, tau[n - 1], W0, W1, work);

    for (idx_t j = 0; j < n - 1; j++)
        W(j, n - 1) = 0.;
    scal(-tau[n - 1], v);
    W(n - 1, n - 1) = T(1) - tau[n - 1];

    for (idx_t i = n - 1; i-- > 0;) {
        auto v = slice(W, range{n, m + n}, i);
        auto W0 = slice(W, i, range{i + 1, k});
        auto W1 = slice(W, range{n, m + n}, range{i + 1, k});

        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v, tau[i], W0, W1, work);

        for (idx_t j = 0; j < i; j++)
            W(j, i) = 0.;
        scal(-tau[i], v);
        W(i, i) = T(1) - tau[i];
    }
}

#endif  // TLAPACK_TRGE_UNG2R_HH