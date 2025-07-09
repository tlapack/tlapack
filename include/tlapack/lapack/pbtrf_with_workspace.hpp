/// @file trmm_out.hpp
/// @author Ella Addison-Taylor, Kyle Cunningham, University of Colorado Denver,
/// USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_PBTRF_WITH_WORKSPACE_HH
#define TLAPACK_PBTRF_WITH_WORKSPACE_HH

#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/potf2.hpp>
#include <tlapack/lapack/mult_llh.hpp>
#include <tlapack/lapack/mult_uhu.hpp>

namespace tlapack {

/// @brief Options struct for pbtrf_with_workspace()
struct BlockedBandedCholeskyOpts : public EcOpts {
    constexpr BlockedBandedCholeskyOpts(const EcOpts& opts = {})
        : EcOpts(opts) {};

    size_t nb = 32;  // Block size
};
/** 
 * 
 * Cholesky factorization of a full, banded, n by n matrix. 
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite band being assumed to be zero:
 *     - Uplo::Lower: A is lower banded.
 *     - Uplo::Upper: A is upper banded.
 *     - Uplo::General: A would be treated as lower banded.
 *
 * @param[in, out] A
 *     A banded, full, n by n Hermitian matrix
 *
 * @param[in] kd The band size.
 *
 * @param[in] opts Options.
 *    - Define the size of the block, nb.
 *    - Default is nb = 32.
 *
 * @ingroup auxiliary
 **/

template<typename uplo_t, typename matrix_t>
void pbtrf_with_workspace(uplo_t uplo, matrix_t& A, size_t kd, const BlockedBandedCholeskyOpts& opts = {})
 {
     using T = tlapack::type_t<matrix_t>;
     using idx_t = tlapack::size_type<matrix_t>;
     using range = tlapack::pair<idx_t, idx_t>;
     using real_t = tlapack::real_type<T>;

     tlapack::Create<matrix_t> new_matrix;

     const idx_t nb = opts.nb;

     idx_t n = nrows(A);

     std::vector<T> work_(nb * nb);
     for (idx_t ii = 0; ii < nb * nb; ++ii) {
         if constexpr (tlapack::is_complex<T>) {
             work_[ii] = T(0, 0);
         }
         else
             work_[ii] = 0;
     }
     auto work = new_matrix(work_, nb, nb);

     if (uplo == tlapack::Uplo::Upper) {
         for (idx_t i = 0; i < n; i += nb) {
             idx_t ib = (n < nb + i) ? n - i : nb;

             auto A00 = slice(A, range(i, min(ib + i, n)),
                                  range(i, std::min(i + ib, n)));

             potf2(tlapack::Uplo::Upper, A00);

             if (i + ib < n) {
                 // i2 = min(kd - ib, n - i - ib)
                 idx_t i2 = (kd + i < n) ? kd - ib : n - i - ib;
                 // i3 = min(ib, n-i-kd)
                 idx_t i3 = (n > i + kd) ? min(ib, n - i - kd) : 0;

                 if (i2 > 0) {
                     auto A01 =
                         slice(A, range(i, ib + i),
                                   range(i + ib, std::min(i + ib + i2, n)));

                     trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                          tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                          real_t(1), A00, A01);

                     auto A11 = slice(A, range(i + ib, std::min(i + kd, n)),
                                          range(i + ib, std::min(i + kd, n)));

                     herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                          real_t(-1), A01, real_t(1), A11);
                 }

                 if (i3 > 0) {
                     auto A02 = slice(A, range(i, i + ib),
                                          range(i + kd, i + kd + i3));

                     auto work02 = slice(work, range(0, ib), range(0, i3));

                     for (idx_t jj = 0; jj < i3; jj++)
                         for (idx_t ii = jj; ii < ib; ++ii)
                             work02(ii, jj) = A02(ii, jj);

                     trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                          tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                          real_t(1), A00, work02);

                     auto A12 =
                         slice(A, range(i + ib, i + kd),
                                   range(i + kd, std::min(i + kd + i3, n)));

                     auto A01 =
                         slice(A, range(i, ib + i),
                                   range(i + ib, std::min(i + ib + i2, n)));

                     gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                          real_t(-1), A01, work02, real_t(1), A12);

                     auto A22 =
                         slice(A, range(i + kd, std::min(i + kd + i3, n)),
                                   range(i + kd, std::min(i + kd + i3, n)));

                     herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                          real_t(-1), work02, real_t(1), A22);

                     for (idx_t jj = 0; jj < i3; ++jj) {
                         for (idx_t ii = jj; ii < ib; ++ii) {
                             A02(ii, jj) = work02(ii, jj);
                         }
                     }
                 }
             }
         }
     }
     else {  // uplo == Lower

         for (idx_t i = 0; i < n; i += nb) {
             idx_t ib = (nb + i < n) ? ib = nb : n - i;

             auto A00 =
                 slice(A, range(i, i + ib), range(i, std::min(ib + i, n)));

             potf2(tlapack::Uplo::Lower, A00);

             if (i + ib <= n) {
                 // i2 = min(kd - ib, n - i - ib)
                 idx_t i2 = (kd + i < n) ? kd - ib : n - i - ib;
                 // i3 = min(ib, n-i-kd)
                 idx_t i3 = (n > i + kd) ? min(ib, n - i - kd) : 0;

                 if (i2 > 0) {
                     auto A10 = slice(A, range(ib + i, ib + i2 + i),
                                          range(i, ib + i));

                     trsm(tlapack::Side::Right, tlapack::Uplo::Lower,
                          tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                          real_t(1), A00, A10);

                     auto A11 = slice(A, range(ib + i, ib + i2 + i),
                                          range(i + ib, i + ib + i2));

                     herk(uplo, tlapack::Op::NoTrans, real_t(-1), A10,
                          real_t(1), A11);
                 }

                 if (i3 > 0) {
                     auto A10 = slice(A, range(ib + i, ib + i2 + i),
                                          range(i, ib + i));

                     auto A20 = slice(A, range(kd + i, min(kd + i3 + i, n)),
                                          range(i, i + ib));

                     auto work20 = slice(work, range(0, i3), range(0, ib));

                     for (idx_t jj = 0; jj < ib; jj++) {
                         idx_t iiend = min(jj + 1, i3);
                         for (idx_t ii = 0; ii < iiend; ++ii) {
                             work20(ii, jj) = A20(ii, jj);
                         }
                     }

                     trsm(tlapack::Side::Right, uplo, tlapack::Op::ConjTrans,
                          tlapack::Diag::NonUnit, real_t(1), A00, work20);

                     auto A21 = slice(A, range(kd + i, kd + i + i3),
                                          range(i + ib, i + ib + i2));

                     gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans,
                          real_t(-1), work20, A10, real_t(1), A21);

                     auto A22 = slice(A, range(kd + i, kd + i + i3),
                                          range(kd + i, kd + i + i3));

                     herk(uplo, tlapack::Op::NoTrans, real_t(-1), work20,
                          real_t(1), A22);

                     for (idx_t jj = 0; jj < ib; jj++) {
                         idx_t iiend = min(jj + 1, i3);
                         for (idx_t ii = 0; ii < iiend; ++ii) {
                             A20(ii, jj) = work20(ii, jj);
                         }
                     }
                 }
             }
         }
     }
 }

 }  // namespace tlapack

#endif  // TLAPACK_PBTRF_WITH_WORKSPACE_HH