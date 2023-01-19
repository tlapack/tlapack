/// @file her.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_HER_HH
#define TLAPACK_BLAS_HER_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Hermitian matrix rank-1 update:
 * \[
 *     A := \alpha x x^H + A,
 * \]
 * where alpha is a real scalar, x is a vector,
 * and A is an n-by-n Hermitian matrix.
 * 
 * Mind that if alpha is complex, the output matrix is no longer Hermitian.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed from symmetry.
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] alpha Real scalar.
 * @param[in] x A n-element vector.
 * @param[in,out] A A n-by-n Hermitian matrix.
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 *
 * @ingroup blas2
 */
template< class matrixA_t, class vectorX_t, class alpha_t,
    enable_if_t<(
    /* Requires: */
        ! is_complex<alpha_t>::value
    ), int > = 0,
    class T = type_t<matrixA_t>,
    disable_if_allow_optblas_t<
        pair< alpha_t, real_type<T> >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >
    > = 0
>
void her(
    Uplo uplo,
    const alpha_t& alpha,
    const vectorX_t& x,
    matrixA_t& A )
{
    // data traits
    using idx_t = size_type< matrixA_t >;
    using scalar_t = scalar_type< alpha_t, type_t<vectorX_t> >;

    // constants
    const idx_t n = nrows(A);

    // check arguments
    tlapack_check_false( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    tlapack_check_false( size(x)  != n );
    tlapack_check_false( ncols(A) != n );


    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            const scalar_t tmp = alpha * conj( x[j] );
            for (idx_t i = 0; i < j; ++i)
                A(i,j) += x[i] * tmp;
            A(j,j) = real( A(j,j) ) + real(x[j])*real(tmp)
                                    - imag(x[j])*imag(tmp);
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            const scalar_t tmp = alpha * conj( x[j] );
            A(j,j) = real( A(j,j) ) + real(x[j])*real(tmp)
                                    - imag(x[j])*imag(tmp);
            for (idx_t i = j+1; i < n; ++i)
                A(i,j) += x[i] * tmp;
        }
    }
}

#ifdef USE_LAPACKPP_WRAPPERS

    template<
        class matrixA_t,
        class vectorX_t,
        class alpha_t,
        class T = type_t<matrixA_t>,
        enable_if_allow_optblas_t<
            pair< alpha_t, real_type<T> >,
            pair< matrixA_t, T >,
            pair< vectorX_t, T >
        > = 0
    >
    inline
    void her(
        Uplo  uplo,
        const alpha_t alpha,
        const vectorX_t& x,
        matrixA_t& A )
    {
        using idx_t = size_type< matrixA_t >;

        // Legacy objects
        auto A_ = legacy_matrix(A);
        auto x_ = legacy_vector(x);

        // Constants to forward
        const idx_t& n = A_.n;
        const idx_t incx = (x_.direction == Direction::Forward) ? idx_t(x_.inc) : idx_t(-x_.inc);
        
        return ::blas::her(
            (::blas::Layout) A_.layout,
            (::blas::Uplo) uplo,
            n,
            alpha,
            x_.ptr, incx,
            A_.ptr, A_.ldim );
    }

#endif

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_HER_HH
