/// @file schur_move.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dtrexc.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_SCHUR_MOVE_HH
#define TLAPACK_SCHUR_MOVE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/schur_swap.hpp"

namespace tlapack
{

    /** schur_move reorders the Schur factorization of a matrix
     *  S = Q*A*Q**H, so that the diagonal element of T with row index IFST
     *  is moved to row ILST.
     *
     *
     * @return  0 if success
     * @return  1 two adjacent blocks were too close to swap (the problem
                  is very ill-conditioned); T may have been partially
                  reordered, and ILST points to the first row of the
                  current position of the block being moved.
     *
     * @param[in]     want_q bool
     *                Whether or not to apply the transformations to Q
     * @param[in,out] A n-by-n matrix.
     *                Must be in Schur form
     * @param[in,out] Q n-by-n matrix.
     *                Orthogonal matrix, not referenced if want_q is false
     * @param[in,out] ifst integer
     *                Initial row index of the eigenvalue block
     * @param[in,out] ilst integer
     *                Index of the eigenvalue block at the exit of the routine.
     *                If it is not possible to move the eigenvalue block to the
     *                given ilst, the value may be changed.
     *
     * @ingroup auxiliary
     */
    template < typename matrix_t >
    int schur_move(bool want_q, matrix_t &A, matrix_t &Q, size_type<matrix_t> &ifst, size_type<matrix_t> &ilst)
    {
        using idx_t = size_type<matrix_t>;
        using T = type_t<matrix_t>;
        using real_t = real_type<T>;

        const idx_t n = ncols(A);
        const real_t zero(0);

        // Quick return
        if (n == 0)
            return 0;

        // Check if ifst points to the middle of a 2x2 block
        if (!is_complex<T>::value)
            if (ifst > 0)
                if (A(ifst, ifst - 1) != zero)
                    ifst = ifst - 1;

        // Size of the current block, can be either 1, 2
        idx_t nbf = 1;
        if (!is_complex<T>::value)
            if (ifst < n - 1)
                if (A(ifst + 1, ifst) != zero)
                    nbf = 2;

        // Check if ilst points to the middle of a 2x2 block
        if (!is_complex<T>::value)
            if (ilst > 0)
                if (A(ilst, ilst - 1) != zero)
                    ilst = ilst - 1;

        // Size of the final block, can be either 1, 2
        idx_t nbl = 1;
        if (!is_complex<T>::value)
            if (ilst < n - 1)
                if (A(ilst + 1, ilst) != zero)
                    nbl = 2;

        idx_t here = ifst;
        if (ifst < ilst)
        {
            if( nbf == 2 and nbl == 1 )
                ilst = ilst - 1;
            if( nbf == 1 and nbl == 2 )
                ilst = ilst + 1;


            while (here != ilst)
            {
                // Size of the next eigenvalue block
                idx_t nbnext = 1;
                if (!is_complex<T>::value)
                    if (here + nbf + 1 < n)
                        if (A(here + nbf + 1, here + nbf) != zero)
                            nbnext = 2;

                int ierr = schur_swap(want_q, A, Q, here, nbf, nbnext);
                if (ierr)
                {
                    // The swap failed, return with error
                    ilst = here;
                    return 1;
                }
                here = here + nbnext;
            }
        }
        else
        {
            while (here != ilst)
            {
                // Size of the next eigenvalue block
                idx_t nbnext = 1;
                if (!is_complex<T>::value)
                    if (here > 1)
                        if (A(here - 1, here - 2) != zero)
                            nbnext = 2;

                int ierr = schur_swap(want_q, A, Q, here - nbnext, nbnext, nbf);
                if (ierr)
                {
                    // The swap failed, return with error
                    ilst = here;
                    return 1;
                }
                here = here - nbnext;
            }
        }

        return 0;
    }

} // lapack

#endif // TLAPACK_SCHUR_MOVE_HH
