/// @file larft3_recursive.hpp Forms the triangular factor T of a block
/// reflector.
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
/// @author Nate Tebeje, Metropolitan State University of Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/larft.h
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LARFT3_RECURSIVE_HH
#define TLAPACK_LARFT3_RECURSIVE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/blas/trmv.hpp"
#include "tlapack/lapack/lacpy.hpp"

namespace tlapack {

/** Forms the triangular factor T of a block reflector H of order n,
 * which is defined as a product of k elementary reflectors.
 *
 * If direction = Direction::Forward,  H = H_1 H_2 ... H_k and T is upper
 * triangular. If direction = Direction::Backward, H = H_k ... H_2 H_1 and T is
 * lower triangular.
 *
 * If storeMode = StoreV::Columnwise, the vector which defines the elementary
 * reflector H(i) is stored in the i-th column of the array V, and
 *
 *               H  =  I - V * T * V'
 *
 * The shape of the matrix V and the storage of the vectors which define
 * the H(i) is best illustrated by the following example with n = 5 and
 * k = 3. The elements equal to 1 are not stored. The rest of the
 * array is not used.
 *
 *     direction = Forward and          direction = Forward and
 *     storeMode = Columnwise:             storeMode = Rowwise:
 *
 *     V = (  1       )                 V = (  1 v1 v1 v1 v1 )
 *         ( v1  1    )                     (     1 v2 v2 v2 )
 *         ( v1 v2  1 )                     (        1 v3 v3 )
 *         ( v1 v2 v3 )
 *         ( v1 v2 v3 )
 *
 *     direction = Backward and         direction = Backward and
 *     storeMode = Columnwise:             storeMode = Rowwise:
 *
 *     V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
 *         ( v1 v2 v3 )                     ( v2 v2 v2  1    )
 *         (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
 *         (     1 v3 )
 *         (        1 )
 *
 * @tparam direction_t Either Direction or any class that implements `operator
 * Direction()`.
 * @tparam storage_t Either StoreV or any class that implements `operator
 * StoreV()`.
 *
 * @param[in] direction
 *     Indicates how H is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $H = H(1) H(2) ... H(k)$.
 *     - Direction::Backward: $H = H(k) ... H(2) H(1)$.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise: n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:    k-by-n matrix V.
 *
 * @param[out] Tmatrix Matrix of size k-by-k containing the triangular factors
 *      of the block reflector.
 *     - Direction::Forward:  T is upper triangular.
 *     - Direction::Backward: T is lower triangular.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixT_t>
int larft3_recursive(direction_t direction,
                     storage_t storeMode,
                     const matrixV_t& V,
                     matrixT_t& Tmatrix)
{
    // data traits
    using std::size_t;
    using idx_t = size_type<matrixV_t>;

    using T = type_t<matrixV_t>;

    // using
    using range = pair<idx_t, idx_t>;

    // constant
    const idx_t n = (storeMode == StoreV::Columnwise) ? nrows(V) : ncols(V);
    const idx_t k = (storeMode == StoreV::Columnwise) ? ncols(V) : nrows(V);
    ;

    // check arguments
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);
    tlapack_check_false(storeMode != StoreV::Columnwise &&
                        storeMode != StoreV::Rowwise);
    tlapack_check_false(
        k > ((storeMode == StoreV::Columnwise) ? ncols(V) : nrows(V)));
    tlapack_check_false(nrows(Tmatrix) < k || ncols(Tmatrix) < k);

    // Quick return
    if (n == 0 || k == 0) {
        return 0;
        // base case
    }
    else if (n == 1 || k == 1) {
        return 0;
    }

    bool qr =
        (direction == Direction::Forward && storeMode == StoreV::Columnwise);

    bool lq = (direction == Direction::Forward && storeMode == StoreV::Rowwise);

    bool ql =
        (direction == Direction::Backward && storeMode == StoreV::Columnwise);

    bool rq =
        (direction == Direction::Backward && storeMode == StoreV::Rowwise);

    if (qr) {
        const idx_t l = k / 2;

        auto V0 = slice(V, range(0, n), range(0, l));
        auto V1 = slice(V, range(l, n), range(l, k));

        auto T00 = slice(Tmatrix, range(0, l), range(0, l));
        auto T01 = slice(Tmatrix, range(0, l), range(l, k));
        auto T11 = slice(Tmatrix, range(l, k), range(l, k));

        larft3_recursive(direction, storeMode, V0, T00);
        larft3_recursive(direction, storeMode, V1, T11);

        for (idx_t j = 0; j < l; ++j) {
            for (idx_t i = 0; i < k - l; ++i) {
                T01(j, i) = conj(V(l + i, j));
            }
        }

        auto V11 = slice(V, range(l, k), range(l, k));

        trmm(Side::Right, Uplo::Lower, Op::NoTrans, Diag::Unit, T(1), V11, T01);

        if (n > k) {
            auto V20 = slice(V, range(k, n), range(0, l));
            auto V21 = slice(V, range(k, n), range(l, k));

            gemm(Op::ConjTrans, Op::NoTrans, T(1), V20, V21, T(1), T01);
        }
        trmm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(-1), T00,
             T01);

        // T01 = T01 * T11
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1), T11,
             T01);
    }
    else if (lq) {
        const idx_t l = k / 2;

        auto v0 = slice(V, range(0, l), range(0, n));
        auto v1 = slice(V, range(l, k), range(l, n));

        auto T00 = slice(Tmatrix, range(0, l), range(0, l));
        auto T01 = slice(Tmatrix, range(0, l), range(l, k));
        auto T11 = slice(Tmatrix, range(l, k), range(l, k));

        larft3_recursive(direction, storeMode, v0, T00);

        larft3_recursive(direction, storeMode, v1, T11);

        auto V01 = slice(V, range(0, l), range(l, k));
        lacpy(Uplo::General, V01, T01);

        auto V11 = slice(V, range(l, k), range(l, k));
        trmm(Side::Right, Uplo::Upper, Op::ConjTrans, Diag::Unit, T(1), V11,
             T01);

        if (n > k) {
            auto V02 = slice(V, range(0, l), range(k, n));
            auto V12 = slice(V, range(l, k), range(k, n));

            gemm(Op::NoTrans, Op::ConjTrans, T(1), V02, V12, T(1), T01);
        }

        trmm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(-1), T00,
             T01);

        // T01 = T01 * T11
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1), T11,
             T01);
    }
    else if (ql) {
        const idx_t l = k / 2;

        auto v0 = slice(V, range(0, l), range(0, n));
        auto v1 = slice(V, range(l, k), range(l, n));

        auto T00 = slice(Tmatrix, range(0, l), range(0, l));
        auto T01 = slice(Tmatrix, range(0, l), range(l, k));
        auto T11 = slice(Tmatrix, range(l, k), range(l, k));

        larft3_recursive(direction, storeMode, v0, T00);

        larft3_recursive(direction, storeMode, v1, T11);

        for (idx_t j = 0; j < k - l; ++j) {
            for (idx_t i = 0; i < l; ++i) {
                Tmatrix(k - l + i, j) = V(n - k + j, k - l + i);
            }
        }
        auto V11 = slice(V, range(l, k), range(l, k));
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::Unit, T(1), V11, T01);

        if (n > k) {
            auto V02 = slice(V, range(0, l), range(k, n));
            auto V12 = slice(V, range(l, k), range(k, n));

            gemm(Op::Trans, Op::NoTrans, T(1), V02, V12, T(1), T01);
        }

        trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, T(-1), T00,
             T01);

        // T01 = T01 * T11
        trmm(Side::Right, Uplo::Lower, Op::NoTrans, Diag::NonUnit, T(1), T11,
             T01);
        // rq case
    }
    else {
        const idx_t l = k / 2;

        auto v0 = slice(V, range(0, l), range(0, n));
        auto v1 = slice(V, range(l, k), range(l, n));

        auto T00 = slice(Tmatrix, range(0, l), range(0, l));
        auto T01 = slice(Tmatrix, range(0, l), range(l, k));
        auto T11 = slice(Tmatrix, range(l, k), range(l, k));

        larft3_recursive(direction, storeMode, v0, T00);

        larft3_recursive(direction, storeMode, v1, T11);

        auto V01 = slice(V, range(0, l), range(l, k));
        lacpy(Uplo::General, V01, T01);

        auto V11 = slice(V, range(l, k), range(l, k));
        trmm(Side::Right, Uplo::Lower, Op::Trans, Diag::Unit, T(1), V11, T01);

        if (n > k) {
            auto V02 = slice(V, range(0, l), range(k, n));
            auto V12 = slice(V, range(l, k), range(k, n));

            gemm(Op::NoTrans, Op::ConjTrans, T(1), V02, V12, T(1), T01);
        }

        trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, T(-1), T00,
             T01);

        // T01 = T01 * T11
        trmm(Side::Right, Uplo::Lower, Op::NoTrans, Diag::NonUnit, T(1), T11,
             T01);
    }
    return 0;
}

/**
 * @tparam direction_t Either Direction or any class that implements `operator
 * Direction()`.
 * @tparam storage_t Either StoreV or any class that implements `operator
 * StoreV()`.
 *
 * @param[in] direction
 *     Indicates how H is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $H = H(1) H(2) ... H(k)$.
 *     - Direction::Backward: $H = H(k) ... H(2) H(1)$.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise: n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:    k-by-n matrix V.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_SMATRIX matrixV_t>
int larft3_recursive(direction_t direction, storage_t storeMode, matrixV_t& V)
{
    // data traits
    using std::size_t;
    using idx_t = size_type<matrixV_t>;

    using T = type_t<matrixV_t>;

    // using
    using range = pair<idx_t, idx_t>;

    // constant
    const idx_t n = (storeMode == StoreV::Columnwise) ? nrows(V) : ncols(V);
    const idx_t k = (storeMode == StoreV::Columnwise) ? ncols(V) : nrows(V);

    // Upper Triangle of V, which holds the T matrix
    auto Tmatrix = slice(V, range{0, k}, range{0, k});

    // check arguments
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);
    tlapack_check_false(storeMode != StoreV::Columnwise &&
                        storeMode != StoreV::Rowwise);
    tlapack_check_false(
        k > ((storeMode == StoreV::Columnwise) ? ncols(V) : nrows(V)));
    tlapack_check_false(nrows(Tmatrix) < k || ncols(Tmatrix) < k);

    // Quick return
    if (n == 0 || k == 0) {
        return 0;
        // base case
    }
    else if (n == 1 || k == 1) {
        return 0;
    }

    bool qr =
        (direction == Direction::Forward && storeMode == StoreV::Columnwise);

    bool lq = (direction == Direction::Forward && storeMode == StoreV::Rowwise);

    bool ql =
        (direction == Direction::Backward && storeMode == StoreV::Columnwise);

    bool rq =
        (direction == Direction::Backward && storeMode == StoreV::Rowwise);

    if (qr) {
        const idx_t l = k / 2;

        auto V0 = slice(V, range(0, n), range(0, l));
        auto V1 = slice(V, range(l, n), range(l, k));

        auto T00 = slice(Tmatrix, range(0, l), range(0, l));
        auto T01 = slice(Tmatrix, range(0, l), range(l, k));
        auto T11 = slice(Tmatrix, range(l, k), range(l, k));

        larft3_recursive(direction, storeMode, V0, T00);
        larft3_recursive(direction, storeMode, V1, T11);

        for (idx_t j = 0; j < l; ++j) {
            for (idx_t i = 0; i < k - l; ++i) {
                T01(j, i) = conj(V(l + i, j));
            }
        }

        auto V11 = slice(V, range(l, k), range(l, k));

        trmm(Side::Right, Uplo::Lower, Op::NoTrans, Diag::Unit, T(1), V11, T01);

        if (n > k) {
            auto V20 = slice(V, range(k, n), range(0, l));
            auto V21 = slice(V, range(k, n), range(l, k));

            gemm(Op::ConjTrans, Op::NoTrans, T(1), V20, V21, T(1), T01);
        }
        trmm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(-1), T00,
             T01);

        // T01 = T01 * T11
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1), T11,
             T01);
    }
    else if (lq) {
        const idx_t l = k / 2;

        auto v0 = slice(V, range(0, l), range(0, n));
        auto v1 = slice(V, range(l, k), range(l, n));

        auto T00 = slice(Tmatrix, range(0, l), range(0, l));
        auto T01 = slice(Tmatrix, range(0, l), range(l, k));
        auto T11 = slice(Tmatrix, range(l, k), range(l, k));

        larft3_recursive(direction, storeMode, v0, T00);

        larft3_recursive(direction, storeMode, v1, T11);

        auto V01 = slice(V, range(0, l), range(l, k));
        lacpy(Uplo::General, V01, T01);

        auto V11 = slice(V, range(l, k), range(l, k));
        trmm(Side::Right, Uplo::Upper, Op::ConjTrans, Diag::Unit, T(1), V11,
             T01);

        if (n > k) {
            auto V02 = slice(V, range(0, l), range(k, n));
            auto V12 = slice(V, range(l, k), range(k, n));

            gemm(Op::NoTrans, Op::ConjTrans, T(1), V02, V12, T(1), T01);
        }

        trmm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(-1), T00,
             T01);

        // T01 = T01 * T11
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1), T11,
             T01);
    }
    else if (ql) {
        const idx_t l = k / 2;

        auto v0 = slice(V, range(0, l), range(0, n));
        auto v1 = slice(V, range(l, k), range(l, n));

        auto T00 = slice(Tmatrix, range(0, l), range(0, l));
        auto T01 = slice(Tmatrix, range(0, l), range(l, k));
        auto T11 = slice(Tmatrix, range(l, k), range(l, k));

        larft3_recursive(direction, storeMode, v0, T00);

        larft3_recursive(direction, storeMode, v1, T11);

        for (idx_t j = 0; j < k - l; ++j) {
            for (idx_t i = 0; i < l; ++i) {
                Tmatrix(k - l + i, j) = V(n - k + j, k - l + i);
            }
        }
        auto V11 = slice(V, range(l, k), range(l, k));
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::Unit, T(1), V11, T01);

        if (n > k) {
            auto V02 = slice(V, range(0, l), range(k, n));
            auto V12 = slice(V, range(l, k), range(k, n));

            gemm(Op::Trans, Op::NoTrans, T(1), V02, V12, T(1), T01);
        }

        trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, T(-1), T00,
             T01);

        // T01 = T01 * T11
        trmm(Side::Right, Uplo::Lower, Op::NoTrans, Diag::NonUnit, T(1), T11,
             T01);
        // rq case
    }
    else {
        const idx_t l = k / 2;

        auto v0 = slice(V, range(0, l), range(0, n));
        auto v1 = slice(V, range(l, k), range(l, n));

        auto T00 = slice(Tmatrix, range(0, l), range(0, l));
        auto T01 = slice(Tmatrix, range(0, l), range(l, k));
        auto T11 = slice(Tmatrix, range(l, k), range(l, k));

        larft3_recursive(direction, storeMode, v0, T00);

        larft3_recursive(direction, storeMode, v1, T11);

        auto V01 = slice(V, range(0, l), range(l, k));
        lacpy(Uplo::General, V01, T01);

        auto V11 = slice(V, range(l, k), range(l, k));
        trmm(Side::Right, Uplo::Lower, Op::Trans, Diag::Unit, T(1), V11, T01);

        if (n > k) {
            auto V02 = slice(V, range(0, l), range(k, n));
            auto V12 = slice(V, range(l, k), range(k, n));

            gemm(Op::NoTrans, Op::ConjTrans, T(1), V02, V12, T(1), T01);
        }

        trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, T(-1), T00,
             T01);

        // T01 = T01 * T11
        trmm(Side::Right, Uplo::Lower, Op::NoTrans, Diag::NonUnit, T(1), T11,
             T01);
    }
    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LARFT3_RECURSIVE_HH
