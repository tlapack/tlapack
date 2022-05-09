/// @file gehrd.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dgehrd.f
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_GEHRD_HH__
#define __TLAPACK_GEHRD_HH__

#include "legacy_api/base/utils.hpp"
#include "legacy_api/base/types.hpp"
#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/lahr2.hpp"

#include <memory>

namespace tlapack
{

    /**
     * Options struct for gehrd
     */
    template <typename idx_t, typename T>
    struct gehrd_opts_t {
        // Blocksize used in the blocked reduction
        idx_t nb = 64;
        // If only nx_switch columns are left, the algorithm will use unblocked code
        idx_t nx_switch = 2;
        // Workspace pointer, if no workspace is provided, one will be allocated internally
        T* _work=nullptr;
        // Workspace size
        idx_t lwork;
    };

    /**
     * Returns the required workspace for gehrd.
     * The arguments are the same as for gehrd itself.
     * 
     * @return idx_t The size of the required workspace
     */
    template <class matrix_t, class vector_t, typename idx_t = size_type<matrix_t>, typename TA = type_t<matrix_t>>
    idx_t get_work_gehrd(size_type<matrix_t> ilo, size_type<matrix_t> ihi, matrix_t &A, vector_t &tau, const gehrd_opts_t<idx_t, TA> &opts = {})
    {
        const idx_t n = ncols(A);
        idx_t nb = opts.nb;

        return (n+nb)*nb;
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
     * @return -i if the ith argument is invalid
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
     * @param[in,out] opts Struct containing the options
     *      See gehrd_opts_t for more details
     *
     * @ingroup gehrd
     */
    template <class matrix_t, class vector_t, typename idx_t = size_type<matrix_t>, typename TA = type_t<matrix_t>>
    int gehrd(size_type<matrix_t> ilo, size_type<matrix_t> ihi, matrix_t &A, vector_t &tau, const gehrd_opts_t<idx_t, TA> &opts = {})
    {
        using pair = pair<idx_t, idx_t>;
        using internal::colmajor_matrix;

        // constants
        const TA one(1);
        const idx_t n = ncols(A);

        // Blocksize
        idx_t nb = opts.nb;
        // Size of the last block which be handled with unblocked code
        idx_t nx_switch = opts.nx_switch;
        idx_t nx = std::max( nb, nx_switch );

        // check arguments
        lapack_error_if((ilo < 0) or (ilo >= n), -1);
        lapack_error_if((ihi < 0) or (ihi > n), -2);
        lapack_error_if(access_denied(dense, write_policy(A)), -3);
        lapack_error_if(ncols(A) != nrows(A), -3);
        lapack_error_if((idx_t)size(tau) < n - 1, -4);

        // quick return
        if (n <= 0)
            return 0;

        // Get the workspace
        TA* _work;
        idx_t lwork;
        idx_t required_workspace = get_work_gehrd(ilo, ihi, A, tau, opts);
        // Store whether or not a workspace was locally allocated
        bool locally_allocated = false;
        if( opts._work and required_workspace <= opts.lwork ){
            // Provided workspace is large enough, use it
            _work = opts._work;
            lwork = opts.lwork;
        } else {
            // No workspace provided or not large enough, allocate it
            locally_allocated = true;
            lwork = required_workspace;
            _work = new TA[lwork];
        }


        auto Y = colmajor_matrix<TA>(&_work[0], n, nb);
        auto T = colmajor_matrix<TA>(&_work[n*nb], nb, nb);

        idx_t i = ilo;
        for (; i+nx < ihi-1; i = i + nb)
        {
            auto nb2 = std::min(nb, ihi - i - 1);

            auto V = slice(A, pair{i + 1, ihi}, pair{i, i + nb2});
            auto A2 = slice(A, pair{0, ihi}, pair{i, ihi});
            auto tau2 = slice(tau, pair{i, ihi});
            auto T_s = slice(T, pair{0, nb2}, pair{0, nb2});
            auto Y_s = slice(Y, pair{0, n}, pair{0, nb2});
            lahr2(i, nb2, A2, tau2, T_s, Y_s);
            if( i + nb2 < ihi ){
                // Note, this V2 contains the last row of the triangular part
                auto V2 = slice(V, pair{nb2-1, ihi-i-1}, pair{0, nb2});

                // Apply the block reflector H to A(0:ihi,i+nb:ihi) from the right, computing
                // A := A - Y * V**T. The multiplication requires V(nb2-1,nb2-1) to be set to 1.
                auto ei = V(nb2-1, nb2-1);
                V(nb2-1, nb2-1) = one;
                auto A3 = slice(A, pair{0, ihi}, pair{i + nb2, ihi});
                auto Y_2 = slice(Y, pair{0, ihi}, pair{0, nb2});
                gemm(Op::NoTrans, Op::ConjTrans, -one, Y_2, V2, one, A3);
                V(nb2-1, nb2-1) = ei;
            }
            // Apply the block reflector H to A(0:i+1,i+1:i+ib) from the right
            auto V1 = slice(A, pair{i + 1, i + nb2 + 1}, pair{i, i + nb2});
            trmm(Side::Right, Uplo::Lower, Op::ConjTrans, Diag::Unit, one, V1, Y_s);
            for (idx_t j = 0; j < nb2 - 1; ++j)
            {
                auto A4 = slice(A, pair{0, i+1}, i + j + 1);
                axpy(-one, slice(Y, pair{0, i+1}, j), A4);
            }

            // Apply the block reflector H to A(i+1:ihi,i+nb:n) from the left
            auto A5 = slice(A, pair{i + 1, ihi}, pair{i + nb2, n});
            auto Y_left = colmajor_matrix<TA>(&_work[0], nb2, n - i - nb2, nb2);
            larfb(Side::Left, Op::ConjTrans, Direction::Forward, StoreV::Columnwise, V, T_s, A5, Y_left);
        }

        auto workspace_vector = col( Y, 0 );
        gehd2( i, ihi, A, tau, workspace_vector );

        if(locally_allocated)
            delete _work;
        return 0;
    }

} // lapack

#endif // __GEHRD_HH__
