/// @file hetd2.hpp
/// @author Skylar Johns, University of Colorado Denver, USA
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HETD2_HH
#define TLAPACK_HETD2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/dot.hpp"
#include "tlapack/blas/hemv.hpp"
#include "tlapack/blas/her2.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Reduce a hermitian matrix to real symmetric tridiagonal form.
 *
 * @return  0 if success
 *
 * @tparam uplo_t Either Uplo or any class that implements `operator Uplo()`.
 *
 * @param[in] uplo
 *      - Uplo::Upper:   Upper triangle of A is referenced;
 *      - Uplo::Lower:   Lower triangle of A is referenced;
 *
 * @param[in,out] A n-by-n symmetric matrix.
 *      On exit, the main diagonal and offdiagonal contain the elements of the
 * symmetric tridiagonal matrix B. The other positions are used to store
 * elementary Householder reflectors.
 *
 * @param[out] tau Vector of length n-1.
 *      The scalar factors of the elementary reflectors.
 *
 * @ingroup computational
 */
template <class matrix_t, class vector_t, class uplo_t>
int hetd2(uplo_t uplo, matrix_t& A, vector_t& tau)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(A);
    const real_t one(1);
    const real_t zero(0);
    const real_t half(0.5);

    // check arguments
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check(nrows(A) == ncols(A));
    tlapack_check((idx_t)size(tau) >= n - 1);

    // quick return
    if (n <= 0) return 0;

    if (uplo == Uplo::Upper) {
        //
        // Reduce upper triangle of A
        //
        A(n - 1, n - 1) = real(A(n - 1, n - 1));
        for (idx_t i = n - 2; i != idx_t(-1); --i) {
            // Define v := A[0:i,i+1]
            auto v = slice(A, pair{0, i}, i + 1);

            // Generate elementary reflector H(i) = I - tau * v * v**T
            // to annihilate A(0:i-1,i+1)
            T taui;
            larfg(backward, columnwise_storage, v, taui);

            if (taui != zero) {
                // Apply H(i) from both sides to A(0:i, 0:i)
                auto C = slice(A, pair{0, i}, pair{0, i});
                auto w = slice(tau, pair{0, i});

                // Store the offdiagonal element
                auto beta = A(i, i + 1);
                A(i, i + 1) = one;

                // Compute x:= taui * C * v storing x in w
                hemv(upperTriangle, taui, C, v, w);

                // Compute w := w - (1/2) * tau * (w**H * v) * v
                axpy(-half * taui * dot(w, v), v, w);

                // Apply the transformation as a rank-2 update:
                //    C := C - v * w**H - w * v**H
                her2(upperTriangle, -one, v, w, C);

                // Reload the offdiagonal element
                A(i, i + 1) = beta;
            }
            else {
                A(i, i) = real(A(i, i));
            }

            tau[i] = taui;
        }
    }
    else {
        //
        // Reduce lower triangle of A
        //
        A(0, 0) = real(A(0, 0));
        for (idx_t i = 0; i < n - 1; ++i) {
            // Define v := A[i+1:n,i]
            auto v = slice(A, pair{i + 1, n}, i);

            // Generate elementary reflector H(i) = I - tau * v * v**T
            // to annihilate A(i+2:n,i)
            T taui;
            larfg(forward, columnwise_storage, v, taui);

            if (taui != zero && i + 1 < n) {
                // Apply H(i) from both sides to A(i+1:n, i+1:n)
                auto C = slice(A, pair{i + 1, n}, pair{i + 1, n});
                auto w = slice(tau, pair{i, n - 1});

                // Store the offdiagonal element
                auto beta = A(i + 1, i);
                A(i + 1, i) = one;

                // Compute x:= taui * C * v storing x in w
                hemv(lowerTriangle, taui, C, v, w);

                // Compute w := w - (1/2) * tau * (w**H * v) * v
                axpy(-half * taui * dot(w, v), v, w);

                // Apply the transformation as a rank-2 update:
                //    C := C - v * w**H - w * v**H
                her2(lowerTriangle, -one, v, w, C);

                // Reload the offdiagonal element
                A(i + 1, i) = beta;
            }
            else {
                A(i + 1, i + 1) = real(A(i + 1, i + 1));
            }

            tau[i] = taui;
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GEQR2_HH
