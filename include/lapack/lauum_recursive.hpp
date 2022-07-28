/// @file lauum_recursive.hpp
/// @author Heidi Meier, University of Colorado Denver
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAUUM_RECURSIVETLAPACK_
#define TLAPACK_LAUUM_RECURSIVETLAPACK_

#include "base/utils.hpp"
#include "base/types.hpp"

namespace tlapack
{

    /** LAUUM is a specific type of inplace HERK. Given `A` a triangular 
     * matrix (lower or upper), LAUUM computes the Hermitian matrix 
     * `upper times lower`. 
     * 
     * If `A` is lower triangular, then LAUUM computes `A^H * A`. If `A` 
     * is upper triangular in input, then LAUUM computes `A*A^H`. The output 
     * (symmetric) matrix is stored in place of the input triangular matrix.
     * 
     * This is the recursive variant.
     *
     * @param[in] uplo
     *      - Uplo::Upper: Upper triangle of `A` is referenced; the strictly lower
     *      triangular part of `A` is not referenced.
     *      - Uplo::Lower: Lower triangle of `A` is referenced; the strictly upper
     *      triangular part of `A` is not referenced.
     *
     * @param[in,out] A n-by-n (upper of lower) (triangular or symmetric) matrix.
     *      On entry, the (upper of lower) part of the n-by-n triangular matrix.
     *      On exit, the (upper of lower) part of the n-by-n symmetric matrix `A^H * A` or `A * A^H`.
     * 
     * @return = 0: successful exit
     *
     * @todo: implement nx to bail out of recursion before 1-by-1 case
     *
     */
    template <typename matrix_t>
    int lauum_recursive(const Uplo &uplo, matrix_t &C)

    {
        tlapack_check(nrows(C) == ncols(C));

        using T = type_t<matrix_t>;
        using idx_t = size_type<matrix_t>;
        using range = std::pair<idx_t, idx_t>;
        using real_t = real_type<T>;

        const idx_t n = nrows(C);

        // check arguments
        tlapack_check_false(uplo != Uplo::Lower &&
                                uplo != Uplo::Upper );
        tlapack_check_false(access_denied(uplo, write_policy(C)) );
        tlapack_check_false(nrows(C) != ncols(C) );

        // Quick return
        if (n <= 0)
            return 0;

        idx_t n0 = n / 2;

        // 1-by-1 case for recursion
        if (n == 1)
        {

            real_t rC00 = real(C(0, 0));
            real_t iC00 = imag(C(0, 0));
            C(0, 0) = make_scalar<T>( rC00*rC00 + iC00*iC00, 0 );

        }
        else
        {
            if (uplo == Uplo::Lower)
            {
                // Upper computes U * U_hermitian
                auto C00 = slice(C, range(0, n0), range(0, n0));
                auto C10 = slice(C, range(n0, n), range(0, n0));
                auto C11 = slice(C, range(n0, n), range(n0, n));

                lauum_recursive(uplo, C00);
                herk(Uplo::Lower, Op::ConjTrans, real_t(1.0), C10, real_t(1.0), C00);
                trmm(Side::Left, uplo, Op::ConjTrans, Diag::NonUnit, real_t(1.0), C11, C10);
                lauum_recursive(uplo, C11);

            }
            else
            {
                // Lower computes  L_hermitian * L
                auto C00 = slice(C, range(0, n0), range(0, n0));
                auto C01 = slice(C, range(0, n0), range(n0, n));
                auto C11 = slice(C, range(n0, n), range(n0, n));

                lauum_recursive(uplo, C00);
                herk(Uplo::Upper, Op::NoTrans, real_t(1.0), C01, real_t(1.0), C00);
                trmm(Side::Right, uplo, Op::ConjTrans, Diag::NonUnit, real_t(1.0), C11, C01);
                lauum_recursive(Uplo::Upper, C11);

            }
        }

        return 0;

    }

} // namespace tlapack

#endif // TLAPACK_LAUUM_RECURSIVETLAPACK_
