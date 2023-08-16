/// @file gen_householder_q.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEN_HOUSEHOLDER_Q_HH
#define TLAPACK_GEN_HOUSEHOLDER_Q_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/ungq.hpp"
#include "tlapack/lapack/ungq_level2.hpp"

namespace tlapack {

enum class GenHouseholderQVariant : char { Level2 = '2', Blocked = 'B' };

struct GenHouseholderQOpts : public UngqOpts {
    GenHouseholderQVariant variant = GenHouseholderQVariant::Blocked;
};

template <class T,
          TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
constexpr WorkInfo gen_householder_q_worksize(
    direction_t direction,
    storage_t storeMode,
    const matrix_t& A,
    const vector_t& tau,
    const GenHouseholderQOpts& opts = {})
{
    if (opts.variant == GenHouseholderQVariant::Level2)
        return ungq_level2_worksize<T>(direction, storeMode, A, tau);
    else
        return ungq_worksize<T>(direction, storeMode, A, tau, opts);
}

template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t>
int gen_householder_q(direction_t direction,
                      storage_t storeMode,
                      matrix_t& A,
                      const vector_t& tau,
                      const GenHouseholderQOpts& opts = {})
{
    if (opts.variant == GenHouseholderQVariant::Level2)
        return ungq_level2(direction, storeMode, A, tau);
    else
        return ungq(direction, storeMode, A, tau, opts);
}

}  // namespace tlapack

#endif