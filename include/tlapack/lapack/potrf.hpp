/// @file potrf.hpp Computes the Cholesky factorization of a Hermitian positive
/// definite matrix A.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_POTRF_HH
#define TLAPACK_POTRF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/potrf2.hpp"
#include "tlapack/lapack/potrf_blocked.hpp"

namespace tlapack {

enum class PotrfVariant : char { Blocked = 'B', Recursive = 'R' };

template <typename idx_t>
struct potrf_opts_t : public ec_opts_t {
    inline constexpr potrf_opts_t(const ec_opts_t& opts = {})
        : ec_opts_t(opts){};

    idx_t nb = 32;  ///< Block size

    PotrfVariant variant = PotrfVariant::Blocked;
};

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A.
 *
 * The factorization has the form
 *      $A = U^H U,$ if uplo = Upper, or
 *      $A = L L^H,$ if uplo = Lower,
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * @tparam uplo_t
 *      Access type: Upper or Lower.
 *      Either Uplo or any class that implements `operator Uplo()`.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *      On entry, the Hermitian matrix A of size n-by-n.
 *
 *      - If uplo = Uplo::Upper, the strictly lower
 *      triangular part of A is not referenced.
 *
 *      - If uplo = Uplo::Lower, the strictly upper
 *      triangular part of A is not referenced.
 *
 *      - On successful exit, the factor U or L from the Cholesky
 *      factorization $A = U^H U$ or $A = L L^H.$
 *
 * @param[in] opts Options.
 *      Define the behavior of checks for NaNs, and nb for potrf_blocked.
 *      - variant:
 *          - Recursive = 'R',
 *          - Blocked = 'B'
 *
 * @return 0: successful exit.
 * @return i, 0 < i <= n, if the leading minor of order i is not
 *      positive definite, and the factorization could not be completed.
 *
 * @ingroup computational
 */
template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t>
inline int potrf(uplo_t uplo,
                 matrix_t& A,
                 const potrf_opts_t<size_type<matrix_t> >& opts = {})
{
    // check arguments
    tlapack_check(uplo == Uplo::Lower || uplo == Uplo::Upper);
    tlapack_check(nrows(A) == ncols(A));
    tlapack_check(opts.variant == PotrfVariant::Blocked ||
                  opts.variant == PotrfVariant::Recursive);

    // Call variant
    if (opts.variant == PotrfVariant::Blocked)
        return potrf_blocked(uplo, A, opts);
    else  // if( opts.variant == PotrfVariant::Recursive )
        return potrf2(uplo, A, opts);
}

}  // namespace tlapack

#endif  // TLAPACK_POTRF_HH
