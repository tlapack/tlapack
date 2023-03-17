/// @file eigenvectors.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dtrevc.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_EIGENVECTORS_HH
#define TLAPACK_EIGENVECTORS_HH

#include <functional>

#include "tlapack/base/utils.hpp"

namespace tlapack {

struct eigenvectors_opts_t : public workspace_opts_t<> {
    inline constexpr eigenvectors_opts_t(const workspace_opts_t<>& opts = {})
        : workspace_opts_t<>(opts){};
};

template <class matrix_t,
          class select_t,
          class T_t = matrix_t,
          class Vl_t = matrix_t,
          class Vr_t = matrix_t>
int eigenvectors(char side,
                 char howmny,
                 select_t select,
                 matrix_t T,
                 size_type<matrix_t>& m,
                 eigenvectors_opts_t& opts)
{
    using idx_t = size_type<matrix_t>;
    using pair = std::pair<idx_t, idx_t>;

    const idx_t n = ncols(A);

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_EIGENVECTORS_HH
