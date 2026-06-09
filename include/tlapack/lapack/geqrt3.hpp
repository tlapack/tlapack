/// @file geqrt3.hpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
//
/// Copyright (c) 2026, University of Colorado Denver. All rights reserved.
//
/// This file is part of <T>LAPACK.
/// <T>LAPACK is free software: you can redistribute it and/or modify it under
/// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEQRT3
#define TLAPACK_GEQRT3

#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {
template <class T, TLAPACK_MATRIX matrix_a, TLAPACK_MATRIX matrix_h>

void geqrt3(matrix_a& A, matrix_h& Tmatrix)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

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
        auto a_vector = tlapack::col(A, 0);

        // Populate matrix T with an elementary reflector
        tlapack::larfg(tlapack::Direction::Forward, tlapack::StoreV::Columnwise,
                       a_vector, Tmatrix(0, 0));
    }
    else {
        // Define slice sizes
        auto n1 = n / 2;
        auto n2 = n - n1;
        auto m1 = n1;
        auto m2 = n2 + n1;
        auto m3 = m;

        // slices
        auto A1 = tlapack::slice(A, range(0, m), range(0, n1));
        auto A11 = tlapack::slice(A, range(0, m1), range(0, n1));
        auto A12 = tlapack::slice(A, range(0, m1), range(n1, n));
        auto A21 = tlapack::slice(A, range(m1, m2), range(0, n1));
        auto A22 = tlapack::slice(A, range(m1, m2), range(n1, n));
        auto A22_32 = tlapack::slice(A, range(m1, m3), range(n1, n));
        auto A31 = tlapack::slice(A, range(m2, m3), range(0, n1));
        auto A32 = tlapack::slice(A, range(m2, m3), range(n1, n));
        auto T11 = tlapack::slice(Tmatrix, range(0, n1), range(0, n1));
        auto T12 = tlapack::slice(Tmatrix, range(0, n1), range(n1, n));
        auto T22 = tlapack::slice(Tmatrix, range(n1, n), range(n1, n));

        // Cut down to one leading column
        tlapack::geqrt3<T>(A1, T11);

        // step 2: Copy A12 into T12
        // no additional flops, just copy
        tlapack::lacpy(tlapack::Uplo::General, A12, T12);

        // step 3: A11ᴴ * T12 = T12

        tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower,
                      tlapack::Op::ConjTrans, tlapack::Diag::Unit,
                      static_cast<T>(1.0), A11, T12);

        // step 4: T12 + (A21ᴴ * A22) = T12

        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                      static_cast<T>(1.0), A21, A22, static_cast<T>(1.0), T12);

        // T12 + (A31ᴴ * A32) = T12
        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                      static_cast<T>(1.0), A31, A32, static_cast<T>(1.0), T12);

        // step 5: T11ᴴ * T12 = T12
        tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Upper,
                      tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                      static_cast<T>(1.0), T11, T12);

        // step 6: A22 - (A21 * T12) = A22
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans,
                      static_cast<T>(-1.0), A21, T12, static_cast<T>(1.0), A22);

        // A32 - (A31 * T12) = A32
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans,
                      static_cast<T>(-1.0), A31, T12, static_cast<T>(1.0), A32);

        // step 7: A11 * T12 = T12
        tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower,
                      tlapack::Op::NoTrans, tlapack::Diag::Unit,
                      static_cast<T>(1.0), A11, T12);

        // step 8: A12 - T12 = A12
        for (idx_t j = 0; j < n2; ++j) {
            for (idx_t i = 0; i < m1; ++i) {
                A12(i, j) -= T12(i, j);
            }
        }
        // step 9: Compute the QR factorization of T22
        tlapack::geqrt3<T>(A22_32, T22);

        // step 10: manually compute T12 = A21ᴴ
        for (idx_t j = 0; j < n2; ++j) {
            for (idx_t i = 0; i < m1; ++i) {
                if constexpr (tlapack::is_complex<T>)
                    T12(i, j) = std::conj(A21(j, i));
                else
                    T12(i, j) = A21(j, i);
            }
        }

        // step 11: T12 = T12 * T22ᴴ
        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Lower,
                      tlapack::Op ::NoTrans, tlapack::Diag::Unit,
                      static_cast<T>(1.0), A22, T12);

        // step 12: T12 = T12 + A31ᴴ * A32
        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                      static_cast<T>(1.0), A31, A32, static_cast<T>(1.0), T12);

        // step 13: T12 = T12 * T11
        tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Upper,
                      tlapack::Op::NoTrans, tlapack::Diag::NonUnit,
                      static_cast<T>(-1.0), T11, T12);

        // step 14: T12 = T12 * T22
        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper,
                      tlapack::Op::NoTrans, tlapack::Diag::NonUnit,
                      static_cast<T>(1.0), T22, T12);
    }
}
}  // namespace tlapack

#endif  // TLAPACK_GEQR2_HH