/// @file legacy_api/lapack/lange.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/lange.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LANGE_HH
#define TLAPACK_LEGACY_LANGE_HH

#include "tlapack/lapack/lange.hpp"

namespace tlapack {
namespace legacy {

    /** Calculates the value of the one norm, Frobenius norm, infinity norm, or
     *element of largest absolute value
     *
     * @return Calculated norm value for the specified type.
     *
     * @param normType Type should be specified as follows:
     *
     *     Norm::Max = maximum absolute value over all elements in A.
     *         Note: this is not a consistent matrix norm.
     *     Norm::One = one norm of the matrix A, the maximum value of the sums
     *of each column. Norm::Inf = the infinity norm of the matrix A, the maximum
     *value of the sum of each row. Norm::Fro = the Frobenius norm of the matrix
     *A. This the square root of the sum of the squares of each element in A.
     *
     * @param m Number of rows to be included in the norm. m >= 0
     * @param n Number of columns to be included in the norm. n >= 0
     * @param A matrix size m-by-n.
     * @param lda Column length of the matrix A.  ldA >= m
     *
     * @ingroup legacy_lapack
     **/
    template <class norm_t, typename TA>
    real_type<TA> lange(
        norm_t normType, idx_t m, idx_t n, const TA* A, idx_t lda)
    {
        using internal::create_matrix;

        // check arguments
        tlapack_check_false(normType != Norm::Fro && normType != Norm::Inf &&
                            normType != Norm::Max && normType != Norm::One);

        // quick return
        if (m == 0 || n == 0) return 0;

        // Views
        auto A_ = create_matrix<TA>((TA*)A, m, n, lda);

        return lange(normType, A_);
    }

}  // namespace legacy
}  // namespace tlapack

#endif  // TLAPACK_LEGACY_LANGE_HH
