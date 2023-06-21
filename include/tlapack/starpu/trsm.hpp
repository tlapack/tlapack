/// @file starpu/trsm.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_TRSM_HH
#define TLAPACK_STARPU_TRSM_HH

#include "tlapack/base/types.hpp"
#include "tlapack/starpu/Matrix.hpp"
#include "tlapack/starpu/gemm.hpp"
#include "tlapack/starpu/tasks.hpp"

namespace tlapack {

/// Overload of trsm for starpu::Matrix
template <class TA, class TB, class alpha_t>
void trsm(Side side,
          Uplo uplo,
          Op trans,
          Diag diag,
          const alpha_t& alpha,
          const starpu::Matrix<TA>& A,
          starpu::Matrix<TB>& B)
{
    using starpu::idx_t;

    // constants
    const TB one(1);
    const idx_t m = B.nrows();
    const idx_t n = B.ncols();
    const idx_t nx = B.get_nx();
    const idx_t ny = B.get_ny();

    // quick return
    if (m == 0 || n == 0) return;

    // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                        trans != Op::ConjTrans);
    tlapack_check_false(diag != Diag::NonUnit && diag != Diag::Unit);
    tlapack_check(A.nrows() == A.ncols());
    tlapack_check(A.nrows() == (side == Side::Left ? m : n));
    tlapack_check(A.get_nx() == A.get_ny());
    tlapack_check(A.get_nx() == (side == Side::Left ? nx : ny));

    // Remove const type from A
    auto& A_ = const_cast<starpu::Matrix<TA>&>(A);

    if (side == Side::Left) {
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for (idx_t ix = 0; ix < nx; ++ix) {
                    for (idx_t iy = 0; iy < ny; ++iy) {
                        starpu::insert_task_trsm<TA, TB>(
                            side, uplo, trans, diag, ((ix == 0) ? alpha : one),
                            A_.tile(nx - ix - 1, nx - ix - 1),
                            B.tile(nx - ix - 1, iy));
                    }
                    auto C = B.get_tiles(0, 0, nx - ix - 1, ny);
                    gemm(trans, noTranspose, -alpha,
                         A.get_const_tiles(0, nx - ix - 1, nx - ix - 1, 1),
                         B.get_const_tiles(nx - ix - 1, 0, 1, ny),
                         ((ix == 0) ? alpha : one), C);
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t ix = 0; ix < nx; ++ix) {
                    for (idx_t iy = 0; iy < ny; ++iy) {
                        starpu::insert_task_trsm<TA, TB>(
                            side, uplo, trans, diag, ((ix == 0) ? alpha : one),
                            A_.tile(ix, ix), B.tile(ix, iy));
                    }
                    auto C = B.get_tiles(ix + 1, 0, nx - ix - 1, ny);
                    gemm(trans, noTranspose, -alpha,
                         A.get_const_tiles(ix + 1, ix, nx - ix - 1, 1),
                         B.get_const_tiles(ix, 0, 1, ny),
                         ((ix == 0) ? alpha : one), C);
                }
            }
        }
        else {  // trans == Op::Trans or Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for (idx_t ix = 0; ix < nx; ++ix) {
                    // auto Aii = A.get_const_tiles(ix, ix, 1, 1);
                    for (idx_t iy = 0; iy < ny; ++iy) {
                        starpu::insert_task_trsm<TA, TB>(
                            side, uplo, trans, diag, ((ix == 0) ? alpha : one),
                            A_.tile(ix, ix), B.tile(ix, iy));
                        // auto B_ = B.get_tiles(ix, iy, 1, 1);
                        // trsm<starpu::Matrix<const TA>>(
                        //     side, uplo, trans, diag, ((ix == 0) ? alpha :
                        //     ((ix == 0) ? alpha : one)), Aii, B_);
                    }
                    auto C = B.get_tiles(ix + 1, 0, nx - ix - 1, ny);
                    gemm(trans, noTranspose, -alpha,
                         A.get_const_tiles(ix, ix + 1, 1, nx - ix - 1),
                         B.get_const_tiles(ix, 0, 1, ny),
                         ((ix == 0) ? alpha : one), C);
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t ix = 0; ix < nx; ++ix) {
                    for (idx_t iy = 0; iy < ny; ++iy) {
                        starpu::insert_task_trsm<TA, TB>(
                            side, uplo, trans, diag, ((ix == 0) ? alpha : one),
                            A_.tile(nx - ix - 1, nx - ix - 1),
                            B.tile(nx - ix - 1, iy));
                    }
                    auto C = B.get_tiles(0, 0, nx - ix - 1, ny);
                    gemm(trans, noTranspose, -alpha,
                         A.get_const_tiles(nx - ix - 1, 0, 1, nx - ix - 1),
                         B.get_const_tiles(nx - ix - 1, 0, 1, ny),
                         ((ix == 0) ? alpha : one), C);
                }
            }
        }
    }
    else {  // side == Side::Right
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for (idx_t iy = 0; iy < ny; ++iy) {
                    for (idx_t ix = 0; ix < nx; ++ix) {
                        starpu::insert_task_trsm<TA, TB>(
                            side, uplo, trans, diag, ((iy == 0) ? alpha : one),
                            A_.tile(iy, iy), B.tile(ix, iy));
                    }
                    auto C = B.get_tiles(0, iy + 1, nx, ny - iy - 1);
                    gemm(noTranspose, trans, -alpha,
                         B.get_const_tiles(0, iy, nx, 1),
                         A.get_const_tiles(iy, iy + 1, 1, ny - iy - 1),
                         ((iy == 0) ? alpha : one), C);
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t iy = 0; iy < ny; ++iy) {
                    for (idx_t ix = 0; ix < nx; ++ix) {
                        starpu::insert_task_trsm<TA, TB>(
                            side, uplo, trans, diag, ((iy == 0) ? alpha : one),
                            A_.tile(ny - iy - 1, ny - iy - 1),
                            B.tile(ix, ny - iy - 1));
                    }
                    auto C = B.get_tiles(0, 0, nx, ny - iy - 1);
                    gemm(noTranspose, trans, -alpha,
                         B.get_const_tiles(0, ny - iy - 1, nx, 1),
                         A.get_const_tiles(ny - iy - 1, 0, 1, ny - iy - 1),
                         ((iy == 0) ? alpha : one), C);
                }
            }
        }
        else {  // trans == Op::Trans or Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for (idx_t iy = 0; iy < ny; ++iy) {
                    for (idx_t ix = 0; ix < nx; ++ix) {
                        starpu::insert_task_trsm<TA, TB>(
                            side, uplo, trans, diag, ((iy == 0) ? alpha : one),
                            A_.tile(ny - iy - 1, ny - iy - 1),
                            B.tile(ix, ny - iy - 1));
                    }
                    auto C = B.get_tiles(0, 0, nx, ny - iy - 1);
                    gemm(noTranspose, trans, -alpha,
                         B.get_const_tiles(0, ny - iy - 1, nx, 1),
                         A.get_const_tiles(0, ny - iy - 1, ny - iy - 1, 1),
                         ((iy == 0) ? alpha : one), C);
                }
            }  // uplo == Uplo::Lower
            else {
                for (idx_t iy = 0; iy < ny; ++iy) {
                    for (idx_t ix = 0; ix < nx; ++ix) {
                        starpu::insert_task_trsm<TA, TB>(
                            side, uplo, trans, diag, ((iy == 0) ? alpha : one),
                            A_.tile(iy, iy), B.tile(ix, iy));
                    }
                    auto C = B.get_tiles(0, iy + 1, nx, ny - iy - 1);
                    gemm(noTranspose, trans, -alpha,
                         B.get_const_tiles(0, iy, nx, 1),
                         A.get_const_tiles(iy + 1, iy, ny - iy - 1, 1),
                         ((iy == 0) ? alpha : one), C);
                }
            }
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_STARPU_TRSM_HH