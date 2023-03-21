/// @file testutils.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
///
/// @brief Utility functions for the unit tests
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TESTUTILS_HH
#define TLAPACK_TESTUTILS_HH

// Definitions
#include "testdefinitions.hpp"

// <T>LAPACK
#include <tlapack/base/utils.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/herk.hpp>
#include <tlapack/blas/iamax.hpp>
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/larf.hpp>
#include <tlapack/lapack/larfg.hpp>
#include <tlapack/lapack/laset.hpp>

namespace tlapack {

class rand_generator {
   private:
    const uint64_t a = 6364136223846793005;
    const uint64_t c = 1442695040888963407;
    uint64_t state = 1302;

   public:
    uint32_t min() { return 0; }

    uint32_t max() { return UINT32_MAX; }

    void seed(uint64_t s) { state = s; }

    uint32_t operator()()
    {
        state = state * a + c;
        return state >> 32;
    }
};

template <typename T, enable_if_t<!is_complex<T>::value, bool> = true>
T rand_helper(rand_generator& gen)
{
    return T(static_cast<float>(gen()) / static_cast<float>(gen.max()));
}

template <typename T, enable_if_t<is_complex<T>::value, bool> = true>
T rand_helper(rand_generator& gen)
{
    using real_t = real_type<T>;
    real_t r1(static_cast<float>(gen()) / static_cast<float>(gen.max()));
    real_t r2(static_cast<float>(gen()) / static_cast<float>(gen.max()));
    return complex_type<real_t>(r1, r2);
}

template <typename T, enable_if_t<!is_complex<T>::value, bool> = true>
T rand_helper()
{
    return T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
}

template <typename T, enable_if_t<is_complex<T>::value, bool> = true>
T rand_helper()
{
    using real_t = real_type<T>;
    real_t r1(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    real_t r2(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    return complex_type<real_t>(r1, r2);
}

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
real_type<type_t<matrix_t>> check_orthogonality(matrix_t& Q, matrix_t& res)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    const idx_t m = nrows(Q);
    const idx_t n = ncols(Q);

    tlapack_check(nrows(res) == ncols(res));
    tlapack_check(nrows(res) == min(m, n));

    // res = I
    laset(Uplo::Upper, (T)0.0, (T)1.0, res);
    if (n <= m) {
        // res = Q'Q - I
        herk(Uplo::Upper, Op::ConjTrans, (real_t)1.0, Q, (real_t)-1.0, res);
    }
    else {
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
real_type<type_t<matrix_t>> check_orthogonality(matrix_t& Q)
{
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    const idx_t m = min(nrows(Q), ncols(Q));

    std::vector<T> res_;
    auto res = new_matrix(res_, m, m);
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
real_type<type_t<matrix_t>> check_similarity_transform(
    matrix_t& A, matrix_t& Q, matrix_t& B, matrix_t& res, matrix_t& work)
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
    gemm(Op::ConjTrans, Op::NoTrans, (real_t)1.0, Q, A, work);
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
real_type<type_t<matrix_t>> check_similarity_transform(matrix_t& A,
                                                       matrix_t& Q,
                                                       matrix_t& B)
{
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    const idx_t n = ncols(A);

    std::vector<T> res_;
    auto res = new_matrix(res_, n, n);
    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);

    return check_similarity_transform(A, Q, B, res, work);
}

/** Generate a general m by n matrix by pre- and post-
 *  multiplying a real diagonal matrix D with random unitary matrices:
 *  A = U*D*V.
 *
 * @param[in] d min(m,n) vector
 * @param[out] A m by n matrix
 * @param[in] kl integer
 *            Number of nonzero subdiagonals
 * @param[in] ku integer
 *            Number of nonzero superdiagonals
 *
 * @ingroup auxiliary
 */
template <class matrix_t, class vector_t, class idx_t = size_type<matrix_t>>
void multiply_rand_unitary(vector_t& d, matrix_t& A, idx_t kl, idx_t ku)
{
    using T = type_t<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    const T zero(0);
    const T one(1);

    // Set A to diagonal matrix, with diagonal equal to elements of d
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = zero;
    for (idx_t i = 0; i < std::min(n, m); ++i)
        A(i, i) = d[i];

    // If kl and ku are zero, we don't need to multiply
    if (kl == 0 and ku == 0) return;

    // Pre- and post-multiply A by random unitary matrices
    T tau;
    std::vector<T> v(std::max(n, m));
    std::vector<T> work(std::max(n, m));
    for (idx_t i2 = 0; i2 < std::min(n, m); ++i2) {
        // idx_t i = std::min(n, m) - i2 - 1;
        idx_t i = i2;
        if (i < m) {
            // generate random reflection
            auto v2 = slice(v, pair{0, m - i});
            for (idx_t j = 0; j < m - i; ++j)
                v2[j] = rand_helper<T>();
            tau = rand_helper<T>();
            larfg(Direction::Forward, StoreV::Columnwise, v2, tau);
            // Multiply A(i:m,i:n) by reflection from the left
            auto A2 = slice(A, pair{i, m}, pair{i, n});
            larf(Side::Left, Direction::Forward, StoreV::Columnwise, v2, tau,
                 A2);
        }
        if (i < n) {
            // generate random reflection
            auto v2 = slice(v, pair{0, n - i});
            for (idx_t j = 0; j < n - i; ++j)
                v2[j] = rand_helper<T>();
            tau = rand_helper<T>();
            larfg(Direction::Forward, StoreV::Columnwise, v2, tau);
            // Multiply A(i:m,i:n) by reflection from the right
            auto A2 = slice(A, pair{i, m}, pair{i, n});
            larf(Side::Right, Direction::Forward, StoreV::Columnwise, v2, tau,
                 A2);
        }
    }

    for (idx_t i = 0; i < std::max(m - kl - 1, n - 1 - ku); ++i) {
        if (kl <= ku) {
            // annihilate subdiagonal elements first (necessary if KL = 0)
            if (m > i + kl) {
                auto v2 = slice(A, pair{i + kl, m}, i);
                larfg(Direction::Forward, StoreV::Columnwise, v2, tau);
                if (i + 1 < n) {
                    auto A2 = slice(A, pair{i + kl, m}, pair{i + 1, n});
                    larf(Side::Left, Direction::Forward, StoreV::Columnwise, v2,
                         tau, A2);
                }
                for (idx_t j = 1; j < size(v2); ++j)
                    v2[j] = zero;
            }
            // annihilate superdiagonal elements
            if (n > i + ku) {
                auto v2 = slice(A, i, pair{i + ku, n});
                larfg(Direction::Forward, StoreV::Rowwise, v2, tau);
                if (i + 1 < n) {
                    auto A2 = slice(A, pair{i + 1, m}, pair{i + ku, n});
                    larf(Side::Right, Direction::Forward, StoreV::Rowwise, v2,
                         tau, A2);
                }
                for (idx_t j = 1; j < size(v2); ++j)
                    v2[j] = zero;
            }
        }
        else {
            // annihilate superdiagonal elements first (necessary if KU = 0)
            if (n > i + ku) {
                auto v2 = slice(A, i, pair{i + ku, n});
                larfg(Direction::Forward, StoreV::Rowwise, v2, tau);
                if (i + 1 < n) {
                    auto A2 = slice(A, pair{i + 1, m}, pair{i + ku, n});
                    larf(Side::Right, Direction::Forward, StoreV::Rowwise, v2,
                         tau, A2);
                }
                for (idx_t j = 1; j < size(v2); ++j)
                    v2[j] = zero;
            }
            // annihilate subdiagonal elements
            if (m > i + kl) {
                auto v2 = slice(A, pair{i + kl, m}, i);
                larfg(Direction::Forward, StoreV::Columnwise, v2, tau);
                if (i + 1 < n) {
                    auto A2 = slice(A, pair{i + kl, m}, pair{i + 1, n});
                    larf(Side::Left, Direction::Forward, StoreV::Columnwise, v2,
                         tau, A2);
                }
                for (idx_t j = 1; j < size(v2); ++j)
                    v2[j] = zero;
            }
        }
    }
}

/** Generate a random matrix with its singular values selected according to a
 * few formulas.
 *
 * @param[in,out] d min(m,n) vector
 * @param[in,out] mode integer
 * @param[out] A m by n matrix
 * @param[in] kl integer
 *            Number of nonzero subdiagonals
 * @param[in] ku integer
 *            Number of nonzero superdiagonals
 *
 * @ingroup auxiliary
 */
template <class matrix_t,
          class vector_t,
          class idx_t = size_type<matrix_t>,
          class real_t = real_type<type_t<matrix_t>>>
void generate_with_known_sv(vector_t& d,
                            int mode,
                            real_t cond,
                            real_t norm,
                            matrix_t& A,
                            idx_t kl,
                            idx_t ku)
{
    real_t one(1);

    idx_t n = ncols(A);
    idx_t m = nrows(A);
    idx_t k = std::min(m, n);

    if (mode == 0) {
        // mode = 0 means we just use the provided vector d
    }
    else if (mode == 1) {
        d[0] = one;
        for (idx_t i = 1; i < k; ++i) {
            d[i] = one / cond;
        }
    }
    else if (mode == 2) {
        for (idx_t i = 0; i + 1 < k; ++i) {
            d[i] = one;
        }
        d[k - 1] = one / cond;
    }
    else if (mode == 3) {
        for (idx_t i = 0; i + 1 < k; ++i) {
            d[i] = one;
        }
        d[k - 1] = one / cond;
        // for (idx_t i = 0; i < k; ++i) {
        //     // d[i] = std::pow(cond, (real_t) - (i) / (k - 1));
        // }
    }
    else {
        // TODO
        assert(false);
    }

    if (mode != 0) {
        // Scale d so that the norm of the matrix is equal to norm
        idx_t imax = iamax(d);
        auto dmax = d[imax];
        auto alpha = norm / dmax;
        scal(alpha, d);
    }

    multiply_rand_unitary(d, A, kl, ku);
}

}  // namespace tlapack

#endif  // TLAPACK_TESTUTILS_HH
