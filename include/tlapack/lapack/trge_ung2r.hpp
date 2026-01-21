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
void trge_ung2r(matrix_t& A0, matrix_t& A1, vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    idx_t n = nrows(A0);
    idx_t m = nrows(A1);
    idx_t k = ncols(A1);

    auto init_view_A0 = slice(A0, range{0, n}, range{n, k});
    laset(GENERAL, real_t(0.0), real_t(0.0), init_view_A0);

    auto init_block_view_A1 = slice(A1, range{0, m}, range{n, k});
    laset(GENERAL, real_t(0.0), real_t(1.0), init_block_view_A1);

    auto view_A = slice(A1, range{0, m}, n - 1);
    auto view_A0 = slice(A0, n - 1, range{n, k});
    auto view_A1 = slice(A1, range{0, m}, range{n, k});

    std::vector<T> work_;
    auto work = new_matrix(work_, m + n, 1);

    if (k > n)
        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, view_A, tau[n - 1], view_A0,
                  view_A1, work);

    for (idx_t j = 0; j < n - 1; j++)
        A0(j, n - 1) = 0.;
    scal(-tau[n - 1], view_A);
    A0(n - 1, n - 1) = T(1) - tau[n - 1];

    for (idx_t i = n - 1; i-- > 0;) {
        auto view_A = slice(A1, range{0, m}, i);
        auto view_A0 = slice(A0, i, range{i + 1, k});
        auto view_A1 = slice(A1, range{0, m}, range{i + 1, k});

        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, view_A, tau[i], view_A0,
                  view_A1, work);

        for (idx_t j = 0; j < i; j++)
            A0(j, i) = 0.;
        scal(-tau[i], view_A);
        A0(i, i) = T(1) - tau[i];
    }
}

#endif  // TLAPACK_TRGE_UNG2R_HH