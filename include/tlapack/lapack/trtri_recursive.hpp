/// @file trtri_recursive.hpp
/// @author Heidi Meier, University of Colorado Denver
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TRTRI_RECURSIVE_HH
#define TLAPACK_TRTRI_RECURSIVE_HH

#include "tlapack/base/utils.hpp"

namespace tlapack
{

    /** TRTRI computes the inverse of a triangular matrix in-place
     * Input is a triangular matrix, output is its inverse
     * This is the recursive variant
     *
     * @param[in] uplo
     *      - Uplo::Upper: Upper triangle of A is referenced; the strictly lower
     *      triangular part of A is not referenced.
     *      - Uplo::Lower: Lower triangle of A is referenced; the strictly upper
     *      triangular part of A is not referenced.
     *
     * @param[in] diag
     *     Whether A has a unit or non-unit diagonal:
     *      - Diag::Unit:    A is assumed to be unit triangular.
     *      - Diag::NonUnit: A is not assumed to be unit triangular.
     * @param[in,out] A n-by-n matrix.
     *      On entry, the n-by-n triangular matrix to be inverted.
     *      On exit, the inverse.
     *
     * @param[in] opts Options.
     *
     * @return = 0: successful exit
     * @return = i+1: if A(i,i) is exactly zero.  The triangular
     *          matrix is singular and its inverse can not be computed.
     *
     * @todo: implement nx to bail out of recursion before 1-by-1 case
     *
     */
    template <typename uplo_t, typename matrix_t>
    int trtri_recursive(uplo_t uplo, Diag diag, matrix_t &C, const ec_opts_t &opts = {})
    {

        using T = type_t<matrix_t>;
        using idx_t = size_type<matrix_t>;
        using range = std::pair<idx_t, idx_t>;
        using real_t = real_type<T>;

        const idx_t n = nrows(C);

        const T zero(0.0);

        // check arguments
        tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
        tlapack_check_false( diag != Diag::NonUnit && diag != Diag::Unit );
        tlapack_check_false(access_denied(uplo, write_policy(C)));
        tlapack_check_false(nrows(C) != ncols(C));

        // Quick return
        if (n <= 0)
            return 0;

        idx_t n0 = n / 2;

        if (n == 1)
        {
            if (diag == Diag::NonUnit){
                if (C(0, 0) != zero)
                {
                    C(0, 0) = real_t(1.) / C(0, 0);
                    return 0;
                }
                else
                {
                    tlapack_error_internal(opts.ec, 1,
                                        "A diagonal of entry of triangular matrix is exactly zero.");
                    return 1;
                }
            }
            else{
                return 0;
            }
            
        }
        else
        {

            if (uplo == Uplo::Lower)
            {

                auto C00 = slice(C, range(0, n0), range(0, n0));
                auto C10 = slice(C, range(n0, n), range(0, n0));
                auto C11 = slice(C, range(n0, n), range(n0, n));

                trsm(Side::Right, Uplo::Lower, Op::NoTrans, diag, T(-1), C00, C10);
                trsm(Side::Left, Uplo::Lower, Op::NoTrans, diag, T(+1), C11, C10);
                int info = trtri_recursive(Uplo::Lower, diag, C00, opts);
                
                if (info != 0)
                {
                    tlapack_error_internal(opts.ec, info,
                                           "A diagonal of entry of triangular matrix is exactly zero.");
                    return info;
                }
                info = trtri_recursive(Uplo::Lower, diag, C11, opts);
                if (info == 0)
                    return 0;
                else
                {
                    tlapack_error_internal(opts.ec, info + n0,
                                           "A diagonal of entry of triangular matrix is exactly zero.");
                    return info + n0;
                }

                // there are two variants, the code below also works

                // trtri_recursive( Uplo::Lower, C00);
                // trtri_recursive( Uplo::Lower, C11);
                // trmm(Side::Right, Uplo::Lower, Op::NoTrans, Diag::NonUnit, T(-1), C00, C10);
                // trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, T(+1), C11, C10);
            }
            else
            {

                auto C00 = slice(C, range(0, n0), range(0, n0));
                auto C01 = slice(C, range(0, n0), range(n0, n));
                auto C11 = slice(C, range(n0, n), range(n0, n));

                trsm(Side::Left, Uplo::Upper, Op::NoTrans, diag, T(-1), C00, C01);
                trsm(Side::Right, Uplo::Upper, Op::NoTrans, diag, T(+1), C11, C01);
                int info = trtri_recursive(Uplo::Upper, diag,C00, opts);
                if (info != 0)
                {
                    tlapack_error_internal(opts.ec, info,
                                           "A diagonal of entry of triangular matrix is exactly zero.");
                    return info;
                }
                info = trtri_recursive(Uplo::Upper, diag, C11, opts);
                if (info == 0)
                    return 0;
                else
                {
                    tlapack_error_internal(opts.ec, info + n0,
                                           "A diagonal of entry of triangular matrix is exactly zero.");
                    return info + n0;
                }

                // there are two variants, the code below also works

                // trtri_recursive( C00, Uplo::Upper);
                // trtri_recursive( C11, Uplo::Upper);
                // trmm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(-1), C00, C01);
                // trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(+1), C11, C01);
            }
        }
    }

} // tlapack

#endif // TLAPACK_TRTRI_RECURSIVE_HH
