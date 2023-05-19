/// @file starpu/gemm.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_GEMM_HH
#define TLAPACK_STARPU_GEMM_HH

#include "tlapack/base/types.hpp"
#include "tlapack/starpu/Matrix.hpp"
#include "tlapack/starpu/tasks.hpp"

namespace tlapack {

template <class TA, class TB, class TC, class alpha_t, class beta_t>
void gemm(Op transA,
          Op transB,
          const alpha_t& alpha,
          const starpu::Matrix<TA>& A,
          const starpu::Matrix<TB>& B,
          const beta_t& beta,
          starpu::Matrix<TC>& C)

{
    using starpu::idx_t;

    // constants
    const real_type<TC> one(1);
    const idx_t m = C.nrows();
    const idx_t n = C.ncols();
    const idx_t k = (transA == Op::NoTrans) ? A.ncols() : A.nrows();
    const idx_t nx = C.get_nx();
    const idx_t ny = C.get_ny();
    const idx_t nz = (transA == Op::NoTrans) ? A.get_ny() : A.get_nx();

    // quick return
    if (m == 0 || n == 0) return;
    if (k == 0 && beta == one) return;

    // check arguments
    tlapack_check(k != 0); /// TODO: Implement this case
    tlapack_check(transA == Op::NoTrans || transA == Op::Trans ||
                  transA == Op::ConjTrans);
    tlapack_check(transB == Op::NoTrans || transB == Op::Trans ||
                  transB == Op::ConjTrans);
    tlapack_check(m == (transA == Op::NoTrans ? A.nrows() : A.ncols()));
    tlapack_check(nx == (transA == Op::NoTrans ? A.get_nx() : A.get_ny()));
    tlapack_check(n == (transB == Op::NoTrans ? B.ncols() : B.nrows()));
    tlapack_check(k == (transB == Op::NoTrans ? B.nrows() : B.ncols()));
    tlapack_check(ny == (transB == Op::NoTrans ? B.get_ny() : B.get_nx()));
    tlapack_check(nz == (transB == Op::NoTrans ? B.get_nx() : B.get_ny()));

    // Remove const type from A and B
    auto& A_ = const_cast<starpu::Matrix<TA>&>(A);
    auto& B_ = const_cast<starpu::Matrix<TB>&>(B);

    for (idx_t ix = 0; ix < nx; ++ix) {
        for (idx_t iy = 0; iy < ny; ++iy) {
            if (transA == Op::NoTrans) {
                if (transB == Op::NoTrans) {
                    if (nz > 0)
                        starpu::insert_task_gemm<TA, TB, TC>(
                            transA, transB, alpha, A_.get_tile_handle(ix, 0),
                            B_.get_tile_handle(0, iy), beta,
                            C.get_tile_handle(ix, iy));
                    for (idx_t iz = 1; iz < nz; ++iz)
                        starpu::insert_task_gemm<TA, TB, TC>(
                            transA, transB, alpha, A_.get_tile_handle(ix, iz),
                            B_.get_tile_handle(iz, iy), one,
                            C.get_tile_handle(ix, iy));
                }
                else {
                    if (nz > 0)
                        starpu::insert_task_gemm<TA, TB, TC>(
                            transA, transB, alpha, A_.get_tile_handle(ix, 0),
                            B_.get_tile_handle(iy, 0), beta,
                            C.get_tile_handle(ix, iy));
                    for (idx_t iz = 1; iz < nz; ++iz)
                        starpu::insert_task_gemm<TA, TB, TC>(
                            transA, transB, alpha, A_.get_tile_handle(ix, iz),
                            B_.get_tile_handle(iy, iz), one,
                            C.get_tile_handle(ix, iy));
                }
            }
            else {
                if (transB == Op::NoTrans) {
                    if (nz > 0)
                        starpu::insert_task_gemm<TA, TB, TC>(
                            transA, transB, alpha, A_.get_tile_handle(0, ix),
                            B_.get_tile_handle(0, iy), beta,
                            C.get_tile_handle(ix, iy));
                    for (idx_t iz = 1; iz < nz; ++iz)
                        starpu::insert_task_gemm<TA, TB, TC>(
                            transA, transB, alpha, A_.get_tile_handle(iz, ix),
                            B_.get_tile_handle(iz, iy), one,
                            C.get_tile_handle(ix, iy));
                }
                else {
                    if (nz > 0)
                        starpu::insert_task_gemm<TA, TB, TC>(
                            transA, transB, alpha, A_.get_tile_handle(0, ix),
                            B_.get_tile_handle(iy, 0), beta,
                            C.get_tile_handle(ix, iy));
                    for (idx_t iz = 1; iz < nz; ++iz)
                        starpu::insert_task_gemm<TA, TB, TC>(
                            transA, transB, alpha, A_.get_tile_handle(iz, ix),
                            B_.get_tile_handle(iy, iz), one,
                            C.get_tile_handle(ix, iy));
                }
            }
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_STARPU_GEMM_HH