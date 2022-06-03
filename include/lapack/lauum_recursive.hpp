/// @file lauum_recursive.hpp
/// @author Heidi Meier, University of Colorado Denver
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dgehd2.f
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LAUUM_RECURSIVE__TLAPACK_
#define __TLAPACK_LAUUM_RECURSIVE__TLAPACK_

#include <iostream>

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace tlapack
{

    /** LAUUM computes the 
     * Input is a triangular matrix, output is its inverse
     * This is the recursive variant
     *
     * @param[in] uplo
     *      - Uplo::Upper: Upper triangle of A is referenced; the strictly lower
     *      triangular part of A is not referenced.
     *      - Uplo::Lower: Lower triangle of A is referenced; the strictly upper
     *      triangular part of A is not referenced.
     *
     * @param[in,out] A n-by-n matrix.
     *      On entry, the n-by-n triangular matrix to be inverted.
     *      On exit, the inverse.
     *
     * @param[in] ec Exception handling configuration at runtime.
     *
     * @return = 0: successful exit
     * @return > 0: if return value = i, A(i,i) is exactly zero.  The triangular
     *          matrix is singular and its inverse can not be computed.
     *
     * @todo: implement nx to bail out of recursion before 1-by-1 case
     *
     */

    template <typename matrix_t>
    int lauum_recursive(const Uplo &uplo, matrix_t &C, const ErrorCheck &ec = {})

    {
        bool verbose = true;

        tlapack_check(nrows(C) == ncols(C));

        using T = type_t<matrix_t>;
        using idx_t = size_type<matrix_t>;
        using range = std::pair<idx_t, idx_t>;
        using real_t = real_type<T>;

        const idx_t n = nrows(C);

        const T zero(0.0);

        // check arguments
        tlapack_check_false(uplo != Uplo::Lower &&
                                uplo != Uplo::Upper,
                            -1);
        tlapack_check_false(access_denied(uplo, write_policy(C)), -1);
        tlapack_check_false(nrows(C) != ncols(C), -2);

        // Quick return
        if (n <= 0)
            return 0;

        idx_t n0 = n / 2;

        // 1-by-1 case for recursion
        if (n == 1)
        {
                C(0, 0) = conj(C(0, 0)) * C(0, 0);
                return 0;
        }
        else

        //Upper and Lower cases of LAUUM
        //Upper computes U * U_hermitian
        //Lower computes  L_hermitian * L 
        {
            if (uplo == Uplo::Lower)
            {
                auto C00 = slice(C, range(0, n0), range(0, n0));
                auto C10 = slice(C, range(n0, n), range(0, n0));
                auto C11 = slice(C, range(n0, n), range(n0, n));

                int info = lauum_recursive(uplo, C00, ec);
                if (info != 0)
                {
                    tlapack_error_internal(ec, info,
                                           "A diagonal of entry of triangular matrix is exactly zero.");
                    return info;
                }
                herk(Uplo::Lower, Op::ConjTrans, real_t(1.0), C10, real_t(1.0), C00);
                trmm(Side::Left, uplo, Op::ConjTrans, Diag::NonUnit, T(1), C11, C10);
                info = lauum_recursive(uplo, C11, ec);
                if (info == 0)
                    return 0;
                else
                {
                    tlapack_error_internal(ec, info + n0,
                                           "A diagonal of entry of triangular matrix is exactly zero.");
                    return info + n0;
                }
            }

            else
            {
                auto C00 = slice(C, range(0, n0), range(0, n0));
                auto C01 = slice(C, range(0, n0), range(n0, n));
                auto C11 = slice(C, range(n0, n), range(n0, n));

                int info = lauum_recursive(uplo, C00, ec);
                if (info != 0)
                {
                    tlapack_error_internal(ec, info,
                                           "A diagonal of entry of triangular matrix is exactly zero.");
                    return info;
                }
                herk(Uplo::Upper, Op::NoTrans, real_t(1.0), C01, real_t(1.0), C00);
                trmm(Side::Right, uplo, Op::ConjTrans, Diag::NonUnit, T(1), C11, C01);
                info = lauum_recursive(Uplo::Upper, C11, ec);
                if (info == 0)
                    return 0;
                else
                {
                    tlapack_error_internal(ec, info + n0,
                                           "A diagonal of entry of triangular matrix is exactly zero.");
                    return info + n0;
                }
            }
        }
    }

} // lapack

#endif // __TLAPACK_LAUUM_RECURSIVE__TLAPACK_