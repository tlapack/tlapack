/// @file recursive_bulge_move.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __RECURSIVE_BULGE_MOVE_HH__
#define __RECURSIVE_BULGE_MOVE_HH__

#include <memory>
#include <complex>

#include "legacy_api/base/utils.hpp"
#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/lahqr_shiftcolumn.hpp"
#include "lapack/move_bulge.hpp"

namespace tlapack
{
    /** Recursive bulge pushes shifts present in the pencil down to the edge of the matrix.
     *
     * @param[in,out] A  n by n matrix.
     *      Hessenberg matrix with bulges present in the first size(s) positions.
     *
     * @param[in] s  complex vector.
     *      Vector containing the shifts to be used during the sweep
     *
     * @param[out] Q  n by n matrix.
     *      The orthogonal matrix Q
     *
     * @param[in,out] V    3 by size(s)/2 matrix.
     *      Matrix containing delayed reflectors.
     *
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true>
    void recursive_bulge_move(matrix_t &A, vector_t &s, matrix_t &Q, matrix_t &V)
    {

    }

} // lapack

#endif // __RECURSIVE_BULGE_MOVE_HH__
