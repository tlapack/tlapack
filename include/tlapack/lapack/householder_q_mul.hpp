/// @file householder_q_mul.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HOUSEHOLDER_Q_MUL_HH
#define TLAPACK_HOUSEHOLDER_Q_MUL_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/unmq.hpp"
#include "tlapack/lapack/unmq_level2.hpp"

namespace tlapack {

enum class HouseholderQMulVariant : char { Level2 = '2', Blocked = 'B' };

struct HouseholderQMulOpts : public UnmqOpts {
    HouseholderQMulVariant variant = HouseholderQMulVariant::Blocked;
};

template <class T,
          TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
constexpr WorkInfo householder_q_mul_worksize(
    side_t side,
    trans_t trans,
    direction_t direction,
    storage_t storeMode,
    const matrixV_t& V,
    const vector_t& tau,
    const matrixC_t& C,
    const HouseholderQMulOpts& opts = {})
{
    if (opts.variant == HouseholderQMulVariant::Level2)
        return unmq_level2_worksize<T>(side, trans, direction, storeMode, V,
                                       tau, C);
    else
        return unmq_worksize<T>(side, trans, direction, storeMode, V, tau, C,
                                opts);
}

template <TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_SIDE side_t,
          TLAPACK_OP trans_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
int householder_q_mul(side_t side,
                      trans_t trans,
                      direction_t direction,
                      storage_t storeMode,
                      const matrixV_t& V,
                      const vector_t& tau,
                      matrixC_t& C,
                      const HouseholderQMulOpts& opts = {})
{
    if (opts.variant == HouseholderQMulVariant::Level2)
        return unmq_level2(side, trans, direction, storeMode, V, tau, C);
    else
        return unmq(side, trans, direction, storeMode, V, tau, C, opts);
}

}  // namespace tlapack

#endif