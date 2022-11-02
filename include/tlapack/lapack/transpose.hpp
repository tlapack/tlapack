/// @file transpose.hpp Out of place transpose
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TRANSPOSE_HH
#define TLAPACK_TRANSPOSE_HH

#include "tlapack/base/utils.hpp"

namespace tlapack
{
    template <typename idx_t>
    struct transpose_opts_t {
        // Optimization parameter. Matrices smaller than nx will not
        // be transposed using recursion. Must be at least 2.s
        idx_t nx = 16;
    };

    /**
     *
     * @brief conjugate transpose a matrix A into a matrix B.
     *
     * @param[in] A m-by-n matrix
     *      The matrix to be transposed
     *
     * @param[out] B n-by-m matrix
     *      On exit, B = A**H
     * 
     * @param[in] opts Options.
     *
     * @ingroup util
     */
    template <class matrixA_t, class matrixB_t>
    void conjtranspose(matrixA_t &A, matrixB_t &B, const transpose_opts_t<size_type<matrixA_t>> &opts = {} )
    {
        using idx_t = size_type<matrixA_t>;
        using pair = std::pair<idx_t, idx_t>;

        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        tlapack_check(m == ncols(B));
        tlapack_check(n == nrows(B));
        tlapack_check( opts.nx >= 2 );

        if (min(m, n) <= opts.nx)
        {
            // The matrix is small, use direct method and end recursion
            for (idx_t i = 0; i < m; ++i)
                for (idx_t j = 0; j < n; ++j)
                    B(j, i) = conj(A(i, j));
        }
        else
        {
            // The matrix is large, split into subblocks and use recursion
            const idx_t m1 = m / 2;
            const idx_t n1 = n / 2;

            auto A00 = slice(A, pair(0, m1), pair(0, n1));
            auto A01 = slice(A, pair(0, m1), pair(n1, n));
            auto A10 = slice(A, pair(m1, m), pair(0, n1));
            auto A11 = slice(A, pair(m1, m), pair(n1, n));

            auto B00 = slice(B, pair(0, n1), pair(0, m1));
            auto B01 = slice(B, pair(0, n1), pair(m1, m));
            auto B10 = slice(B, pair(n1, n), pair(0, m1));
            auto B11 = slice(B, pair(n1, n), pair(m1, m));

            conjtranspose(A00, B00, opts);
            conjtranspose(A01, B10, opts);
            conjtranspose(A10, B01, opts);
            conjtranspose(A11, B11, opts);
        }
    }

    /**
     *
     * @brief transpose a matrix A into a matrix B.
     *
     * @param[in] A m-by-n matrix
     *      The matrix to be transposed
     *
     * @param[out] B n-by-m matrix
     *      On exit, B = A**T
     * 
     * @param[in] opts Options.
     *
     * @ingroup util
     */
    template <class matrixA_t, class matrixB_t>
    void transpose(matrixA_t &A, matrixB_t &B, const transpose_opts_t<size_type<matrixA_t>> &opts = {} )
    {
        using idx_t = size_type<matrixA_t>;
        using pair = std::pair<idx_t, idx_t>;

        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        tlapack_check(m == ncols(B));
        tlapack_check(n == nrows(B));
        tlapack_check( opts.nx >= 2 );

        if (min(m, n) <= opts.nx)
        {
            // The matrix is small, use direct method and end recursion
            for (idx_t i = 0; i < m; ++i)
                for (idx_t j = 0; j < n; ++j)
                    B(j, i) = A(i, j);
        }
        else
        {
            // The matrix is large, split into subblocks and use recursion
            const idx_t m1 = m / 2;
            const idx_t n1 = n / 2;

            auto A00 = slice(A, pair(0, m1), pair(0, n1));
            auto A01 = slice(A, pair(0, m1), pair(n1, n));
            auto A10 = slice(A, pair(m1, m), pair(0, n1));
            auto A11 = slice(A, pair(m1, m), pair(n1, n));

            auto B00 = slice(B, pair(0, n1), pair(0, m1));
            auto B01 = slice(B, pair(0, n1), pair(m1, m));
            auto B10 = slice(B, pair(n1, n), pair(0, m1));
            auto B11 = slice(B, pair(n1, n), pair(m1, m));

            transpose(A00, B00, opts);
            transpose(A01, B10, opts);
            transpose(A10, B01, opts);
            transpose(A11, B11, opts);
        }
    }

} // lapack

#endif // TLAPACK_TRANSPOSE_HH
