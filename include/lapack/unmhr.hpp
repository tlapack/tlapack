/// @file unmhr.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zunmhr.f
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __UNMHR_HH__
#define __UNMHR_HH__

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larf.hpp"
#include "lapack/ung2r.hpp"

namespace tlapack
{

    /** Applies unitary matrix Q to a matrix C.
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
     * @param work Vector of size n-1.
     *
     * @ingroup gehrd
     */
    template <
        class matrix_t, class vector_t, class work_t>
    int unmhr(
        Side side,
        Op trans,
        size_type<matrix_t> ilo,
        size_type<matrix_t> ihi,
        matrix_t &A,
        vector_t &tau,
        matrix_t &C,
        work_t &work)
    {
        using idx_t = size_type<matrix_t>;
        using pair = std::pair<idx_t, idx_t>;

        auto A_s = slice(A, pair{ilo + 1, ihi}, pair{ilo, ihi-1});
        auto tau_s = slice(tau, pair{ilo, ihi - 1});
        auto C_s = (side == Side::Left) ? slice(C, pair{ilo + 1, ihi}, pair{0, ncols(C)}) : slice(C, pair{0, nrows(C)}, pair{ilo + 1, ihi});

        unm2r(side, trans, A_s, tau_s, C_s, work);

        return 0;
    }

}

#endif // __UNMHR_HH__
