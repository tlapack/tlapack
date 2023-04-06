/// @file larf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/larf.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LARF_HH
#define TLAPACK_LEGACY_LARF_HH

#include "tlapack/lapack/larf.hpp"

namespace tlapack {

/** Applies an elementary reflector H to a m-by-n matrix C.
 *
 * @see larf( Side side, idx_t m, idx_t n, TV const *v, int_t incv, scalar_type<
 * TV, TC , TW > tau, TC *C, idx_t ldC, TW *work )
 *
 * @ingroup legacy_lapack
 */
template <AbstractSide side_t, typename TV, typename TC>
inline void larf(side_t side,
                 idx_t m,
                 idx_t n,
                 TV const* v,
                 int_t incv,
                 scalar_type<TV, TC> tau,
                 TC* C,
                 idx_t ldC)
{
    using internal::colmajor_matrix;

    // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(m < 0);
    tlapack_check_false(n < 0);
    tlapack_check_false(incv == 0);
    tlapack_check_false(ldC < m);

    // Initialize indexes
    idx_t lenv = ((side == Side::Left) ? m : n);

    // Matrix views
    auto C_ = colmajor_matrix<TC>(C, m, n, ldC);

    tlapack_expr_with_vector(
        v_, TV, lenv, v, incv,
        return larf(side, forward, columnwise_storage, v_, tau, C_));
}

}  // namespace tlapack

#endif  // TLAPACK_LEGACY_LARF_HH
