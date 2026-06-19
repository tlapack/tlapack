/// @file tkhnv.hpp Solves a Tikhonov regularized least squares problem using
/// various decompositions.
/// @author L. Carlos Gutierrez, Julien Langou, University of Colorado Denver,
/// USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tik_bidiag_elden.hpp"
#include "tik_qr.hpp"
#include "tik_svd.hpp"
#include "tlapack/base/utils.hpp"

using namespace tlapack;

/// @brief Variants of the algorithm to compute the decomposition.
enum class TikVariant : char { QR = 'Q', Elden = 'E', SVD = 'S' };

struct TikOpts {
    TikVariant variant = TikVariant::QR;

    constexpr TikOpts(TikVariant v = TikVariant::QR) : variant(v) {}
};

/**
 * @param[in] A is an m-by-n matrix where m >= n.
 * @param[in,out] b
 *      On entry, b is a m-by-k matrix
 *
 *      On exit, b is an m-by-k matrix that stores the solution x in the first
 *      n rows.
 * @param[in] lambda scalar
 *@param[in] opts Options.
 *     Choose the decomposition method for solving a Tikhonov regularized least
 *squares problem.
 *      - variant:
 *          - QR = 'Q',
 *          - Eldén's bidiagonalization = 'E'
 *          - SVD = 'S'
 */

template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t>
void tkhnv(matrixA_t& A, matrixb_t& b, real_t lambda, const TikOpts& opts = {})
{
    tlapack_check(nrows(A) >= ncols(A));
    tlapack_check(nrows(b) == nrows(A));
    tlapack_check(opts.variant == TikVariant::QR ||
                  opts.variant == TikVariant::Elden ||
                  opts.variant == TikVariant::SVD);

    if (opts.variant == TikVariant::QR)
        tik_qr(A, b, lambda);
    else if (opts.variant == TikVariant::Elden)
        tik_bidiag_elden(A, b, lambda);
    else
        tik_svd(A, b, lambda);
}
