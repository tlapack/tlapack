/// @file hasinf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HASINF_HH
#define TLAPACK_HASINF_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Returns true if and only if A has an infinite entry.
 *
 * @tparam uplo_t Type of access inside the algorithm.
 *      Either Uplo or any type that implements
 *          operator Uplo().
 *
 * @param[in] uplo Determines the entries of A that will be checked.
 *      The following access types are allowed:
 *          Uplo::General,
 *          Uplo::UpperHessenberg,
 *          Uplo::LowerHessenberg,
 *          Uplo::Upper,
 *          Uplo::Lower,
 *          Uplo::StrictUpper,
 *          Uplo::StrictLower.
 *
 * @param[in] A matrix.
 *
 * @return true if A has an infinite entry.
 * @return false if A has no infinite entry.
 */
template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t>
bool hasinf(uplo_t uplo, const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    tlapack_check(uplo == Uplo::General || uplo == Uplo::UpperHessenberg ||
                  uplo == Uplo::LowerHessenberg || uplo == Uplo::Upper ||
                  uplo == Uplo::Lower || uplo == Uplo::StrictUpper ||
                  uplo == Uplo::StrictLower);

    if (uplo == Uplo::UpperHessenberg) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j + 2 : m); ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j + 1 : m); ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::StrictUpper) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j : m); ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::LowerHessenberg) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = ((j > 1) ? j - 1 : 0); i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::Lower) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j; i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else if (uplo == Uplo::StrictLower) {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 1; i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
    else  // if ( (Uplo) uplo == Uplo::General )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                if (isinf(A(i, j))) return true;
        return false;
    }
}

/**
 * Returns true if and only if A has an infinite entry.
 *
 * Specific implementation for band access types.
 * @see tlapack::hasinf(uplo_t uplo, const matrix_t& A).
 */
template <TLAPACK_MATRIX matrix_t>
bool hasinf(BandAccess accessType, const matrix_t& A) noexcept
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kl = accessType.lower_bandwidth;
    const idx_t ku = accessType.upper_bandwidth;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = ((j >= ku) ? (j - ku) : 0); i < min(m, j + kl + 1); ++i)
            if (isinf(A(i, j))) return true;
    return false;
}

/**
 * Returns true if and only if x has an infinite entry.
 *
 * @param[in] x vector.
 *
 * @return true if x has an infinite entry.
 * @return false if x has no infinite entry.
 */
template <TLAPACK_VECTOR vector_t>
bool hasinf(const vector_t& x) noexcept
{
    using idx_t = size_type<vector_t>;

    // constants
    const idx_t n = size(x);

    for (idx_t i = 0; i < n; ++i)
        if (isinf(x[i])) return true;
    return false;
}

}  // namespace tlapack

#endif  // TLAPACK_HASINF_HH