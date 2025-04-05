/// @file FrancisOpts.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_FRANCIS_OPTS_HH
#define TLAPACK_FRANCIS_OPTS_HH

#include <cmath>
#include <functional>

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Options struct for multishift_qr().
 */
struct FrancisOpts {
    /// Function that returns the number of shifts to use
    /// for a given matrix size
    std::function<size_t(size_t, size_t)> nshift_recommender =
        [](size_t n, size_t nh) -> size_t {
        if (n < 30) return 2;
        if (n < 60) return 4;
        if (n < 150) return 10;
        if (n < 590) return size_t(n / std::log2(n));
        if (n < 3000) return 64;
        if (n < 6000) return 128;
        return 256;
    };

    /// Function that returns the number of shifts to use
    /// for a given matrix size
    std::function<size_t(size_t, size_t)> deflation_window_recommender =
        [](size_t n, size_t nh) -> size_t {
        if (n < 30) return 2;
        if (n < 60) return 4;
        if (n < 150) return 10;
        if (n < 590) return size_t(n / std::log2(n));
        if (n < 3000) return 96;
        if (n < 6000) return 192;
        return 384;
    };

    // On exit of the routine. Stores the number of times AED and sweep were
    // called And the total number of shifts used.
    int n_aed = 0;           ///< number of times AED was called
    int n_sweep = 0;         ///< number of sweeps used
    int n_shifts_total = 0;  ///< total number of shifts used

    /// Threshold to switch between blocked and unblocked code
    size_t nmin = 75;
    /// Threshold of percent of AED window that must converge to skip a sweep
    size_t nibble = 14;
};

// Forward declarations:

template <class T,
          TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
WorkInfo multishift_qr_worksize(bool want_t,
                                bool want_z,
                                size_type<matrix_t> ilo,
                                size_type<matrix_t> ihi,
                                const matrix_t& A,
                                const vector_t& w,
                                const matrix_t& Z,
                                const FrancisOpts& opts = {});

template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_WORKSPACE work_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
int multishift_qr_work(bool want_t,
                       bool want_z,
                       size_type<matrix_t> ilo,
                       size_type<matrix_t> ihi,
                       matrix_t& A,
                       vector_t& w,
                       matrix_t& Z,
                       work_t& work,
                       FrancisOpts& opts);

template <class T,
          TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
WorkInfo aggressive_early_deflation_worksize(bool want_t,
                                             bool want_z,
                                             size_type<matrix_t> ilo,
                                             size_type<matrix_t> ihi,
                                             size_type<matrix_t> nw,
                                             const matrix_t& A,
                                             const vector_t& s,
                                             const matrix_t& Z,
                                             const size_type<matrix_t>& ns,
                                             const size_type<matrix_t>& nd,
                                             const FrancisOpts& opts = {});

template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_WORKSPACE work_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
void aggressive_early_deflation_work(bool want_t,
                                     bool want_z,
                                     size_type<matrix_t> ilo,
                                     size_type<matrix_t> ihi,
                                     size_type<matrix_t> nw,
                                     matrix_t& A,
                                     vector_t& s,
                                     matrix_t& Z,
                                     size_type<matrix_t>& ns,
                                     size_type<matrix_t>& nd,
                                     work_t& work,
                                     FrancisOpts& opts);

}  // namespace tlapack

#endif  // TLAPACK_FRANCIS_OPTS_HH