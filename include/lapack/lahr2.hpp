/// @file lahqr.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlahqr.f
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LAHR2_HH__
#define __LAHR2_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"
#include "blas/gemv.hpp"

namespace lapack
{

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
     * @return -i if the ith argument is invalid
     *
     * @param[in] k integer
     * @param[in] nb integer
     * @param[in,out] A n-by-n matrix.
     * @param[out] tau Real vector of length n-1.
     *      The scalar factors of the elementary reflectors.
     * @param[in,out] T nb-by-nb matrix.
     * @param[in,out] Y n-by-nb matrix.
     *
     * @ingroup gehrd
     */
    template <class matrix_t, class vector_t, class T_t, class Y_t>
    int lahr2(size_type<matrix_t> k, size_type<matrix_t> nb, matrix_t &A, vector_t &tau, T_t &T, Y_t &Y)
    {
        using TA = type_t<matrix_t>;
        using idx_t = size_type<matrix_t>;
        using pair = std::pair<idx_t, idx_t>;

        using blas::axpy;
        using blas::copy;
        using blas::gemv;
        using blas::scal;
        using blas::trmv;

        // constants
        const TA one(1);
        const TA zero(0);
        const idx_t n = nrows(A);

        // quick return if possible
        if (n <= 1)
            return 0;

        TA ei;
        for (idx_t i = 0; i < nb; ++i)
        {
            if (i > 0)
            {
                //
                // Update A(K+1:N,I), this rest will be updated later via
                // level 3 BLAS.
                //

                //
                // Update I-th column of A - Y * V**T
                // (Application of the reflectors from the right)
                //
                auto Y2 = slice(Y, pair{k + 1, n}, pair{0, i});
                auto Vti = slice(A, k + i - 1, pair{0, i});
                auto b = slice(A, pair{k + 1, n}, i);
                gemv(Op::NoTrans, -one, Y2, Vti, one, b);
                //
                // Apply I - V * T**T * V**T to this column (call it b) from the
                // left, using the last column of T as workspace
                //
                // Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
                //          ( V2 )             ( b2 )
                //
                // where V1 is unit lower triangular
                //
                // w := V1**T * b1
                //
                auto b1 = slice(A, pair{k + 1, k + 1 + i}, i);
                auto V1 = slice(A, pair{k + 1, k + 1 + i}, pair{0, i});
                auto w = slice(T, pair{0, i}, nb - 1);
                copy(b1, w);
                trmv(Uplo::Lower, Op::Trans, Diag::Unit, V1, w);
                //
                // w := w + V2**T * b2
                //
                auto b2 = slice(A, pair{k + 1 + i, n}, i);
                auto V2 = slice(A, pair{k + 1 + i, n}, pair{0, i});
                gemv(Op::Trans, one, V2, b2, one, w);
                //
                // w := T**T * w
                //
                auto T2 = slice(T, pair{0, i}, pair{0, i});
                trmv(Uplo::Upper, Op::Trans, Diag::NonUnit, T2, w);
                //
                // b2 := b2 - V2*w
                //
                gemv(Op::NoTrans, -one, V2, w, one, b2);
                //
                // b1 := b1 - V1*w
                //
                trmv(Uplo::Lower, Op::NoTrans, Diag::Unit, V1, w);
                axpy(-one, w, b1);

                A(k + i, i - 1) = ei;
            }
            auto v = slice(A, pair{k + i + 1, n}, i);
            larfg(v, tau[i]);

            // larf has been edited to not require A(k+i,i) = one
            // this is for thread safety. Since we already modified
            // A(k+i,i) before, this is not required here
            ei = v[0];
            v[0] = one;
            //
            // Compute  Y(K+1:N,I)
            //
            auto A2 = slice(A, pair{k + 1, n}, pair{i + 1, n - k});
            auto y = slice(Y, pair{k + 1, n}, i);
            gemv(Op::NoTrans, one, A2, v, zero, y);
            auto t = slice(T, pair{0, i}, i);
            auto A3 = slice(A, pair{k + i + 1, n}, pair{0, i});
            gemv(Op::ConjTrans, one, A3, v, zero, t);
            auto Y2 = slice(Y, pair{k + 1, n}, pair{0, i});
            gemv(Op::NoTrans, -one, Y2, t, one, y);
            scal(tau[i], y);
            //
            // Compute T(0:I+1,I)
            //
            scal(-tau[i], t);
            auto T2 = slice(T, pair{0, i}, pair{0, i});
            trmv(Uplo::Upper, Op::NoTrans, Diag::NonUnit, T2, t);
            T(i, i) = tau[i];
        }

        return 0;
    }

} // lapack

#endif // __LAHR2_HH__
