/// @file hetrf.hpp Computes the Bunch-Kaufman factorization of a symmetric
/// or Hermitian matrix A.
/// @author Hugh M Kadhem, University of California Berkeley, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HETRF_HH
#define TLAPACK_HETRF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/hetrf_blocked.hpp"

namespace tlapack {

/// @brief Variants of the algorithm to compute the Bunch-Kaufman factorization.
enum class HetrfVariant : char {
    Blocked = 'B'  // blocked Bunch-Kaufman with diagonal pivoting
};

/// @brief Options struct for hetrf()
struct HetrfOpts : public BlockedLDLOpts {
    constexpr HetrfOpts(const EcOpts& opts = {}) : BlockedLDLOpts(opts){};

    HetrfVariant variant = HetrfVariant::Blocked;
};

/** @copybrief hetrf()
 * Workspace is provided as an argument.
 * @copydetails hetrf()
 *
 * @param work Workspace. Use the workspace query to determine the size
 * needed.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_UPLO uplo_t,
          TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR ipiv_t,
          TLAPACK_WORKSPACE work_t>
int hetrf_work(uplo_t uplo,
               matrix_t& A,
               ipiv_t& ipiv,
               work_t& work,
               const HetrfOpts& opts = {})
{
    // check arguments
    tlapack_check(uplo == Uplo::Lower || uplo == Uplo::Upper);
    tlapack_check(nrows(A) == ncols(A));
    tlapack_check(opts.invariant == Op::Trans ||
                  opts.invariant == Op::ConjTrans);
    tlapack_check(opts.variant == HetrfVariant::Blocked);
    // Call variant
    if (opts.variant == HetrfVariant::Blocked)
        return hetrf_blocked_work(uplo, A, ipiv, work, opts);
    else
        return 0;
}

/** Computes the Bunch-Kaufman factorization of a symmetric or Hermitian
 * matrix A.
 *
 * The factorization has the form
 *      $A = U D U^{op},$ if uplo = Upper, or
 *      $A = L D L^{op},$ if uplo = Lower,
 * where U resp. L is a product of permutations and upper resp. lower
 * unit-triangular matrices, and D is symmetric block-diagonal with blocks of
 * size 1 or 2.
 * If opts.invariant = Op::Trans then op=T,
 * and if opts.invariant = Op::ConjTrans then op=H.
 *
 * If uplo =Upper, then
 * $$U = prod_{i=n-1}^1 P_iU_i,$$
 * where either $P_i$ is a transposition on rows $i$ and $ipiv[i]$,
 * $U_i$ is the identity plus a strip above the diagonal of column $i$,
 * and $D$ has a 1-by-1 block at diagonal $i$.
 * or $P_i$ is a transposition on rows $i-1$ and $-ipiv[i]$,
 * $U_i$ is the identity plus a strip above the block diagonal of columns $i$
 * and $i-1$, and $D$ has a 2-by-2 block at the diagonals $i-1,i$.
 *
 *  If uplo = Lower, then the factors follow a similar format but the direction
 * of index increments is flipped; so that $P_1L_1$ is the first factor, rank 2
 * factors use indices $i,i+1$.
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
 *      On entry, the symmetric matrix A of size n-by-n.
 *
 *      - If uplo = Uplo::Upper, the strictly lower
 *      triangular part of A is not referenced.
 *
 *      - If uplo = Uplo::Lower, the strictly upper
 *      triangular part of A is not referenced.
 *
 *      - On successful exit, the factors $D$ and $U$ or $L$ from the
 *      Bunch-Kaufman factorization $A = U D U^T$ or $A = L D L^T,$ stored with
 *      the upper or lower triangular parts of the blocks of $D$ at the same
 *      positions of $U$ or $L$.
 *
 * @param[out] ipiv
 *      - On successful exit, if $P_i$ is a rank 1 pivot, then $ipiv[i]$ is the
 *      index it transposes with $i$; if $P_i$ is a rank 2 pivot,
 *      then $-ipiv[i] = -ipiv[i \pm 1]$ = -piv-1 where piv  is the index it
 * transposes with $i \pm 1$, where the increment is positive for uplo=Upper and
 * negative for uplo=Lower.
 *
 * @param[in] opts Options.
 *      Define the behavior of checks for NaNs, and nb for hetrf_blocked.
 *      - variant:
 *          - Blocked = 'B'
 *
 * @return 0: successful exit.
 * @return i, 0 < i <= n, if $D(i-1,i-1)$ is exactly zero;
 *      the factorization has been completed but D is exactly singular
 *      and division by zero will occur if it is used to solve a system of
 *      equations.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR ipiv_t>
int hetrf(uplo_t uplo, matrix_t& A, ipiv_t& ipiv, const HetrfOpts& opts = {})
{
    // check arguments
    tlapack_check(uplo == Uplo::Lower || uplo == Uplo::Upper);
    tlapack_check(nrows(A) == ncols(A));
    tlapack_check(opts.invariant == Op::Trans ||
                  opts.invariant == Op::ConjTrans);
    tlapack_check(opts.variant == HetrfVariant::Blocked);
    // Call variant
    if (opts.variant == HetrfVariant::Blocked)
        return hetrf_blocked(uplo, A, ipiv, opts);
    else
        return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_HETRF_HH
