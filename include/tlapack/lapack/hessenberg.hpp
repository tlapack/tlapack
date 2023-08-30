/// @file hessenberg.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HESSENBERG_HH
#define TLAPACK_HESSENBERG_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/gehd2.hpp"
#include "tlapack/lapack/gehrd.hpp"

namespace tlapack {

/// @brief Variants of the algorithm to reduce a matrix to upper Hessenberg
/// form.
enum class HessenbergVariant : char { Level2 = '2', Blocked = 'B' };

/// @brief Options struct for hessenberg()
struct HessenbergOpts : public GehrdOpts {
    HessenbergVariant variant = HessenbergVariant::Blocked;
};

/** Workspace query of hessenberg()
 *
 * @param[in] ilo integer
 * @param[in] ihi integer
 * @param[in] A n-by-n matrix.
 * @param[in] tau Vector of length n-1.
 *
 * @param[in] opts Options.
 *      - @c opts.variant: Variant of the algorithm to use.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
constexpr WorkInfo hessenberg_worksize(size_type<matrix_t> ilo,
                                       size_type<matrix_t> ihi,
                                       const matrix_t& A,
                                       const vector_t& tau,
                                       const HessenbergOpts& opts = {})
{
    // Call variant
    if (opts.variant == HessenbergVariant::Level2)
        return gehd2_worksize<T>(ilo, ihi, A, tau);
    else
        return gehrd_worksize<T>(ilo, ihi, A, tau, opts);
}

/** @copybrief hessenberg()
 * Workspace is provided as an argument.
 * @copydetails hessenberg()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int hessenberg_work(size_type<matrix_t> ilo,
                    size_type<matrix_t> ihi,
                    matrix_t& A,
                    vector_t& tau,
                    work_t& work,
                    const HessenbergOpts& opts = {})
{
    // Call variant
    if (opts.variant == HessenbergVariant::Level2)
        return gehd2_work(ilo, ihi, A, tau, work);
    else
        return gehrd_work(ilo, ihi, A, tau, work, opts);
}

/** Reduces a general square matrix to upper Hessenberg form
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_ilo H_ilo+1 ... H_ihi,
 * \]
 * Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i] = 0; v[i+1] = 1,
 * \]
 * with v[i+2] through v[ihi] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * @return  0 if success
 *
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      It is assumed that A is already upper Hessenberg in columns
 *      0:ilo and rows ihi:n and is already upper triangular in
 *      columns ihi+1:n and rows 0:ilo.
 *      0 <= ilo <= ihi <= max(1,n).
 * @param[in,out] A n-by-n matrix.
 *      On entry, the n by n general matrix to be reduced.
 *      On exit, the upper triangle and the first subdiagonal of A
 *      are overwritten with the upper Hessenberg matrix H, and the
 *      elements below the first subdiagonal, with the array TAU,
 *      represent the orthogonal matrix Q as a product of elementary
 *      reflectors. See Further Details.
 * @param[out] tau Real vector of length n-1.
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      - @c opts.variant: Variant of the algorithm to use.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int hessenberg(size_type<matrix_t> ilo,
               size_type<matrix_t> ihi,
               matrix_t& A,
               vector_t& tau,
               const HessenbergOpts& opts = {})
{
    // Call variant
    if (opts.variant == HessenbergVariant::Level2)
        return gehd2(ilo, ihi, A, tau);
    else
        return gehrd(ilo, ihi, A, tau, opts);
}

}  // namespace tlapack

#endif