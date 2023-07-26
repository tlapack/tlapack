/// @file unmhr.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zunmhr.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNMHR_HH
#define TLAPACK_UNMHR_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/ung2r.hpp"
#include "tlapack/lapack/unm2r.hpp"

namespace tlapack {

/** Worspace query of unmhr()
 *
 * @param[in] side Specifies which side op(Q) is to be applied.
 *      - Side::Left:  C := op(Q) C;
 *      - Side::Right: C := C op(Q).
 *
 * @param[in] trans The operation $op(Q)$ to be used:
 *      - Op::NoTrans:      $op(Q) = Q$;
 *      - Op::ConjTrans:    $op(Q) = Q^H$.
 *      Op::Trans is a valid value if the data type of A is real. In this case,
 *      the algorithm treats Op::Trans as Op::ConjTrans.
 *
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      ilo and ihi must have the same values as in the
 *      previous call to gehrd. Q is equal to the unit
 *      matrix except in the submatrix Q(ilo+1:ihi,ilo+1:ihi).
 *      0 <= ilo <= ihi <= max(1,n).
 * @param[in] A n-by-n matrix
 *      Matrix containing orthogonal vectors, as returned by gehrd
 * @param[in] tau Vector of length n-1
 *      Contains the scalar factors of the elementary reflectors.
 *
 * @param[in] C m-by-n matrix.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
inline constexpr WorkInfo unmhr_worksize(Side side,
                                         Op trans,
                                         size_type<matrix_t> ilo,
                                         size_type<matrix_t> ihi,
                                         const matrix_t& A,
                                         const vector_t& tau,
                                         const matrix_t& C)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    auto A_s = slice(A, range{ilo + 1, ihi}, range{ilo, ihi - 1});
    auto tau_s = slice(tau, range{ilo, ihi - 1});
    auto C_s = (side == Side::Left)
                   ? slice(C, range{ilo + 1, ihi}, range{0, ncols(C)})
                   : slice(C, range{0, nrows(C)}, range{ilo + 1, ihi});

    return unm2r_worksize<T>(side, trans, A_s, tau_s, C_s);
}

/** Applies unitary matrix Q to a matrix C.
 *
 * @param[in] side Specifies which side op(Q) is to be applied.
 *      - Side::Left:  C := op(Q) C;
 *      - Side::Right: C := C op(Q).
 *
 * @param[in] trans The operation $op(Q)$ to be used:
 *      - Op::NoTrans:      $op(Q) = Q$;
 *      - Op::ConjTrans:    $op(Q) = Q^H$.
 *      Op::Trans is a valid value if the data type of A is real. In this case,
 *      the algorithm treats Op::Trans as Op::ConjTrans.
 *
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      ilo and ihi must have the same values as in the
 *      previous call to gehrd. Q is equal to the unit
 *      matrix except in the submatrix Q(ilo+1:ihi,ilo+1:ihi).
 *      0 <= ilo <= ihi <= max(1,n).
 * @param[in] A n-by-n matrix
 *      Matrix containing orthogonal vectors, as returned by gehrd
 * @param[in] tau Vector of length n-1
 *      Contains the scalar factors of the elementary reflectors.
 *
 * @param[in,out] C m-by-n matrix.
 *      On exit, C is replaced by one of the following:
 *      - side = Side::Left  & trans = Op::NoTrans:    $C := Q C$;
 *      - side = Side::Right & trans = Op::NoTrans:    $C := C Q$;
 *      - side = Side::Left  & trans = Op::ConjTrans:  $C := C Q^H$;
 *      - side = Side::Right & trans = Op::ConjTrans:  $C := Q^H C$.
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int unmhr_work(Side side,
               Op trans,
               size_type<matrix_t> ilo,
               size_type<matrix_t> ihi,
               const matrix_t& A,
               const vector_t& tau,
               matrix_t& C,
               work_t& work)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    auto A_s = slice(A, range{ilo + 1, ihi}, range{ilo, ihi - 1});
    auto tau_s = slice(tau, range{ilo, ihi - 1});
    auto C_s = (side == Side::Left)
                   ? slice(C, range{ilo + 1, ihi}, range{0, ncols(C)})
                   : slice(C, range{0, nrows(C)}, range{ilo + 1, ihi});

    return unm2r_work(side, trans, A_s, tau_s, C_s, work);
}

/** Applies unitary matrix Q to a matrix C.
 *
 * @param[in] side Specifies which side op(Q) is to be applied.
 *      - Side::Left:  C := op(Q) C;
 *      - Side::Right: C := C op(Q).
 *
 * @param[in] trans The operation $op(Q)$ to be used:
 *      - Op::NoTrans:      $op(Q) = Q$;
 *      - Op::ConjTrans:    $op(Q) = Q^H$.
 *      Op::Trans is a valid value if the data type of A is real. In this case,
 *      the algorithm treats Op::Trans as Op::ConjTrans.
 *
 * @param[in] ilo integer
 * @param[in] ihi integer
 *      ilo and ihi must have the same values as in the
 *      previous call to gehrd. Q is equal to the unit
 *      matrix except in the submatrix Q(ilo+1:ihi,ilo+1:ihi).
 *      0 <= ilo <= ihi <= max(1,n).
 * @param[in] A n-by-n matrix
 *      Matrix containing orthogonal vectors, as returned by gehrd
 * @param[in] tau Vector of length n-1
 *      Contains the scalar factors of the elementary reflectors.
 *
 * @param[in,out] C m-by-n matrix.
 *      On exit, C is replaced by one of the following:
 *      - side = Side::Left  & trans = Op::NoTrans:    $C := Q C$;
 *      - side = Side::Right & trans = Op::NoTrans:    $C := C Q$;
 *      - side = Side::Left  & trans = Op::ConjTrans:  $C := C Q^H$;
 *      - side = Side::Right & trans = Op::ConjTrans:  $C := Q^H C$.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int unmhr(Side side,
          Op trans,
          size_type<matrix_t> ilo,
          size_type<matrix_t> ihi,
          const matrix_t& A,
          const vector_t& tau,
          matrix_t& C)
{
    using T = type_t<matrix_t>;

    // functor
    Create<matrix_t> new_matrix;

    // Allocates workspace
    WorkInfo workinfo = unmhr_worksize<T>(side, trans, ilo, ihi, A, tau, C);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return unmhr_work(side, trans, ilo, ihi, A, tau, C, work);
}
}  // namespace tlapack

#endif  // TLAPACK_UNMHR_HH
