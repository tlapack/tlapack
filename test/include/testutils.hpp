/// @file testutils.hpp
/// @brief Utility functions for the unit tests
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <legacy_api/legacyArray.hpp>
#include <tlapack.hpp>

namespace tlapack
{

    /** Calculates res = Q'*Q - I if m <= n or res = Q*Q' otherwise
     *  Also computes the frobenius norm of res.
     *
     * @return frobenius norm of res
     *
     * @param[in] Q m by n (almost) orthogonal matrix
     * @param[out] res n by n matrix as defined above
     *
     * @ingroup auxiliary
     */
    template <class matrix_t>
    real_type<type_t<matrix_t>> check_orthogonality(matrix_t &Q, matrix_t &res)
    {
        using T = type_t<matrix_t>;
        using real_t = real_type<T>;

        auto m = nrows(Q);
        auto n = ncols(Q);

        tlapack_check(nrows(res) == ncols(res));
        tlapack_check(nrows(res) == min(m, n));

        // res = I
        laset(Uplo::Upper, (T)0.0, (T)1.0, res);
        if (n <= m)
        {
            // res = Q'Q - I
            herk(Uplo::Upper, Op::ConjTrans, (real_t)1.0, Q, (real_t)-1.0, res);
        }
        else
        {
            // res = QQ' - I
            herk(Uplo::Upper, Op::NoTrans, (real_t)1.0, Q, (real_t)-1.0, res);
        }

        // Compute ||res||_F
        return lanhe(frob_norm, Uplo::Upper, res);
    }

    /** Calculates ||Q'*Q - I||_F if m <= n or ||Q*Q' - I||_F otherwise
     *
     * @return frobenius norm of error
     *
     * @param[in] Q m by n (almost) orthogonal matrix
     *
     * @ingroup auxiliary
     */
    template <class matrix_t>
    real_type<type_t<matrix_t>> check_orthogonality(matrix_t &Q)
    {
        using T = type_t<matrix_t>;

        auto m = min(nrows(Q), ncols(Q));

        std::unique_ptr<T[]> res_(new T[m * m]);
        auto res = legacyMatrix<T, layout<matrix_t>>(m, m, &res_[0], m);
        return check_orthogonality(Q, res);
    }

    /** Calculates res = Q'*A*Q - B and the frobenius norm of res
     *
     * @return frobenius norm of res
     *
     * @param[in] A n by n matrix
     * @param[in] Q n by n unitary matrix
     * @param[in] B n by n matrix
     * @param[out] res n by n matrix as defined above
     * @param[out] work n by n workspace matrix
     *
     * @ingroup auxiliary
     */
    template <class matrix_t>
    real_type<type_t<matrix_t>> check_similarity_transform(matrix_t &A, matrix_t &Q, matrix_t &B, matrix_t &res, matrix_t &work)
    {
        using T = type_t<matrix_t>;
        using real_t = real_type<T>;

        tlapack_check(nrows(A) == ncols(A));
        tlapack_check(nrows(Q) == ncols(Q));
        tlapack_check(nrows(B) == ncols(B));
        tlapack_check(nrows(res) == ncols(res));
        tlapack_check(nrows(work) == ncols(work));
        tlapack_check(nrows(A) == nrows(Q));
        tlapack_check(nrows(A) == nrows(B));
        tlapack_check(nrows(A) == nrows(res));
        tlapack_check(nrows(A) == nrows(work));

        // res = Q'*A*Q - B
        lacpy(Uplo::General, B, res);
        gemm(Op::ConjTrans, Op::NoTrans, (real_t)1.0, Q, A, (real_t)0.0, work);
        gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, work, Q, (real_t)-1.0, res);

        // Compute ||res||_F
        return lange(frob_norm, res);
    }

    /** Calculates ||Q'*A*Q - B||
     *
     * @return frobenius norm of res
     *
     * @param[in] A n by n matrix
     * @param[in] Q n by n unitary matrix
     * @param[in] B n by n matrix
     *
     * @ingroup auxiliary
     */
    template <class matrix_t>
    real_type<type_t<matrix_t>> check_similarity_transform(matrix_t &A, matrix_t &Q, matrix_t &B)
    {
        using T = type_t<matrix_t>;

        auto n = ncols(A);

        std::unique_ptr<T[]> res_(new T[n * n]);
        auto res = legacyMatrix<T, layout<matrix_t>>(n, n, &res_[0], n);
        std::unique_ptr<T[]> work_(new T[n * n]);
        auto work = legacyMatrix<T, layout<matrix_t>>(n, n, &work_[0], n);

        return check_similarity_transform(A, Q, B, res, work);
    }

}
