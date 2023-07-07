/// @file starpu/herk.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_HERK_HH
#define TLAPACK_STARPU_HERK_HH

#include "tlapack/base/types.hpp"
#include "tlapack/starpu/Matrix.hpp"
#include "tlapack/starpu/gemm.hpp"
#include "tlapack/starpu/tasks.hpp"

namespace tlapack {

/// Overload of herk for starpu::Matrix
template <class TA, class TC, class alpha_t, class beta_t>
void herk(Uplo uplo,
          Op trans,
          const alpha_t& alpha,
          const starpu::Matrix<TA>& A,
          const beta_t& beta,
          starpu::Matrix<TC>& C)

{
    using starpu::idx_t;

    // constants
    const real_type<TC> one(1);
    const idx_t n = (trans == Op::NoTrans) ? A.nrows() : A.ncols();
    const idx_t k = (trans == Op::NoTrans) ? A.ncols() : A.nrows();
    const idx_t nx = (trans == Op::NoTrans) ? A.get_nx() : A.get_ny();
    const idx_t ny = (trans == Op::NoTrans) ? A.get_ny() : A.get_nx();

    // quick return
    if (n == 0) return;
    if (k == 0 && beta == one) return;

    // check arguments
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper &&
                        uplo != Uplo::General);
    tlapack_check_false(trans != Op::NoTrans && trans != Op::ConjTrans);
    tlapack_check_false(C.nrows() != n);
    tlapack_check_false(C.ncols() != n);
    tlapack_check_false(C.get_nx() != nx);
    tlapack_check_false(C.get_ny() != nx);

    // Remove const type from A and B
    auto& A_ = const_cast<starpu::Matrix<TA>&>(A);

    if (trans == Op::NoTrans) {
        for (idx_t ix = 0; ix < nx; ++ix) {
            // Update diagonal tile of C
            starpu::insert_task_herk<TA, TC>(uplo, trans, alpha, A_.tile(ix, 0),
                                             beta, C.tile(ix, ix));
            for (idx_t iy = 1; iy < ny; ++iy)
                starpu::insert_task_herk<TA, TC>(
                    uplo, trans, alpha, A_.tile(ix, iy), one, C.tile(ix, ix));

            // Update off-diagonal tiles of C
            auto Ai = A.get_const_tiles(ix, 0, 1, ny);
            auto Bi = A.get_const_tiles(ix + 1, 0, nx - ix - 1, ny);
            if (uplo == Uplo::Upper || uplo == Uplo::General) {
                auto Ci = C.get_tiles(ix, ix + 1, 1, nx - ix - 1);
                gemm(NO_TRANS, CONJ_TRANS, alpha, Ai, Bi, beta, Ci);
            }
            if (uplo == Uplo::Lower || uplo == Uplo::General) {
                auto Ci = C.get_tiles(ix + 1, ix, nx - ix - 1, 1);
                gemm(NO_TRANS, CONJ_TRANS, alpha, Bi, Ai, beta, Ci);
            }
        }
    }
    else {  // trans == Op::ConjTrans
        for (idx_t ix = 0; ix < nx; ++ix) {
            // Update diagonal tile of C
            starpu::insert_task_herk<TA, TC>(uplo, trans, alpha, A_.tile(0, ix),
                                             beta, C.tile(ix, ix));
            for (idx_t iy = 1; iy < ny; ++iy)
                starpu::insert_task_herk<TA, TC>(
                    uplo, trans, alpha, A_.tile(iy, ix), one, C.tile(ix, ix));

            // Update off-diagonal tiles of C
            auto Ai = A.get_const_tiles(0, ix, ny, 1);
            auto Bi = A.get_const_tiles(0, ix + 1, ny, nx - ix - 1);
            if (uplo == Uplo::Upper || uplo == Uplo::General) {
                auto Ci = C.get_tiles(ix, ix + 1, 1, nx - ix - 1);
                gemm(CONJ_TRANS, NO_TRANS, alpha, Ai, Bi, beta, Ci);
            }
            if (uplo == Uplo::Lower || uplo == Uplo::General) {
                auto Ci = C.get_tiles(ix + 1, ix, nx - ix - 1, 1);
                gemm(CONJ_TRANS, NO_TRANS, alpha, Bi, Ai, beta, Ci);
            }
        }
    }
}
}  // namespace tlapack

#endif  // TLAPACK_STARPU_HERK_HH