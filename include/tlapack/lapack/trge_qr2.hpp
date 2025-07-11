/// @file trge_qr2.hpp
/// @author Julien Langou, L. Carlos Gutierrez, University of Colorado Denver,
/// USA
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TRGE_QR2_HH
#define TLAPACK_TRGE_QR2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfg.hpp"

using namespace tlapack;

template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
void trge_qr2(matrix_t& A, vector_t& tau)
{
    using real_t = real_type<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_t>;

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    idx_t n = ncols(A);
    idx_t m = nrows(A) - n;

    // Generate a workspace
    std::vector<T> work_;
    auto work = new_matrix(work_, m + n, 1);

    for (idx_t i = 0; i < n - 1; ++i) {
        auto v = slice(A, range{n, m + n}, i);

        larfg(COLUMNWISE_STORAGE, A(i, i), v, tau[i]);

        v = slice(A, range{n, m + n}, i);
        auto C0 = slice(A, i, range{i + 1, n});
        auto C1 = slice(A, range{n, m + n}, range{i + 1, n});

        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v, conj(tau[i]), C0, C1, work);
    }

    auto v = slice(A, range{n, m + n}, n - 1);

    larfg(COLUMNWISE_STORAGE, A(n - 1, n - 1), v, tau[n - 1]);
}
#endif  // TLAPACK_TRGE_QR2_HH