/// @file geqrt3.hpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEQRT3_HH
#define TLAPACK_GEQRT3_HH

#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/lapack/larfg.hpp"

/**
 * Recursive QR factorization using compact WY Householder representation.
 *
 * @param[in,out] A
 *     On entry, the m-by-n matrix to be factorized, where m >= n.
 *     On exit, A contains \(R\) and the Householder vectors.
 *
 * @param[out] Tmatrix
 *     n-by-n matrix containing the triangular factor of the compact WY
 *     representation.
 *
 * @ingroup workspace_query
 */

namespace tlapack {
template <TLAPACK_MATRIX matrix_a, TLAPACK_MATRIX matrix_h>

void geqrt3(matrix_a& A, matrix_h& Tmatrix)
{
    using std::size_t;
    using idx_t = size_type<matrix_a>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_a>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    auto info = 0;
    if (m < n) {
        std::cout << "Error: m < n" << std::endl;
        info = -1;
    }

    if (info != 0) {
        return;
    }

    if (n == 1) {
        // Turn the single column into a vector
        auto a_vector = col(A, 0);

        // Populate matrix T with an elementary reflector
        larfg(Direction::Forward, StoreV::Columnwise, a_vector, Tmatrix(0, 0));
    }
    else {
        // Define slice sizes
        idx_t n1 = n / 2;
        idx_t n2 = n - n1;
        idx_t m1 = n1;
        idx_t m2 = n2 + n1;
        idx_t m3 = m;

        // slices
        auto A1 = slice(A, range(0, m), range(0, n1));
        auto A11 = slice(A, range(0, m1), range(0, n1));
        auto A12 = slice(A, range(0, m1), range(n1, n));
        auto A21 = slice(A, range(m1, m2), range(0, n1));
        auto A22 = slice(A, range(m1, m2), range(n1, n));
        auto A22_32 = slice(A, range(m1, m3), range(n1, n));
        auto A31 = slice(A, range(m2, m3), range(0, n1));
        auto A32 = slice(A, range(m2, m3), range(n1, n));
        auto T11 = slice(Tmatrix, range(0, n1), range(0, n1));
        auto T12 = slice(Tmatrix, range(0, n1), range(n1, n));
        auto T22 = slice(Tmatrix, range(n1, n), range(n1, n));

        // step 1: Compute the QR factorization of A1
        geqrt3(A1, T11);

        // step 2: Copy A12 into T12
        // no additional flops, just copy
        lacpy(Uplo::General, A12, T12);

        // step 3: T12 = A11ᴴ * T12

        trmm(Side::Left, Uplo::Lower, Op::ConjTrans, Diag::Unit, T(1.0), A11,
             T12);

        // step 4: T12 = T12 + (A21ᴴ * A22)

        gemm(Op::ConjTrans, Op::NoTrans, T(1.0), A21, A22, T(1.0), T12);

        // T12 = T12 + (A31ᴴ * A32)
        gemm(Op::ConjTrans, Op::NoTrans, T(1.0), A31, A32, T(1.0), T12);

        // step 5: T12 = T11ᴴ * T12
        trmm(Side::Left, Uplo::Upper, Op::ConjTrans, Diag::NonUnit, T(1.0), T11,
             T12);

        // step 6:  A22 = A22 - (A21 * T12)
        gemm(Op::NoTrans, Op::NoTrans, T(-1.0), A21, T12, T(1.0), A22);

        // A32 = A32 - (A31 * T12)
        gemm(Op::NoTrans, Op::NoTrans, T(-1.0), A31, T12, T(1.0), A32);

        // step 7:T12 = A11 * T12
        trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::Unit, T(1.0), A11,
             T12);

        // step 8: A12 = A12 - T12
        for (idx_t j = 0; j < n2; ++j) {
            for (idx_t i = 0; i < m1; ++i) {
                A12(i, j) -= T12(i, j);
            }
        }
        // step 9: Compute the QR factorization of A22_32
        geqrt3(A22_32, T22);

        // step 10: manually compute T12 = A21ᴴ
        for (idx_t j = 0; j < n2; ++j) {
            for (idx_t i = 0; i < m1; ++i) {
                if constexpr (is_complex<T>)
                    T12(i, j) = std::conj(A21(j, i));
                else
                    T12(i, j) = A21(j, i);
            }
        }

        // step 11: T12 = T12 * T22ᴴ
        trmm(Side::Right, Uplo::Lower, Op::NoTrans, Diag::Unit, T(1.0), A22,
             T12);

        // step 12: T12 = T12 + A31ᴴ * A32
        gemm(Op::ConjTrans, Op::NoTrans, T(1.0), A31, A32, T(1.0), T12);

        // step 13: T12 = T12 * T11
        trmm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(-1.0), T11,
             T12);

        // step 14: T12 = T12 * T22
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1.0), T22,
             T12);
    }
}
}  // namespace tlapack
#endif  // TLAPACK_GEQRT3_HH
