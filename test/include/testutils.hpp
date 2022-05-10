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

    /** Calculates res = Q'*Q - I and the frobenius norm of res
     *
     * @return frobenius norm of res
     *
     * @param[in] Q n by n (almost) orthogonal matrix
     * @param[out] res n by n matrix as defined above
     *
     * @ingroup auxiliary
     */
    template <class matrix_t>
    real_type<type_t<matrix_t>> check_orthogonality(matrix_t &Q, matrix_t &res)
    {
        using T = type_t<matrix_t>;

        // res = I
        laset(Uplo::Upper, (T)0.0, (T)1.0, res);
        // res = Q'Q - I
        herk(Uplo::Upper, Op::ConjTrans, (real_type<T>)1.0, Q, (real_type<T>)-1.0, res);

        // Compute ||res||_F
        return lanhe(frob_norm, Uplo::Upper, res);
    }

    /** Calculates res = Q'*A*Q - B and the frobenius norm of res relative to the norm of A
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

        // res = Q'*A*Q - B
        lacpy(Uplo::General, B, res);
        gemm(Op::ConjTrans, Op::NoTrans, (T)1.0, Q, A, (T)0.0, work);
        gemm(Op::NoTrans, Op::NoTrans, (T)1.0, work, Q, (T)-1.0, res);

        // Compute ||res||_F/||A||_F
        return lange(frob_norm, res);
    }

}
