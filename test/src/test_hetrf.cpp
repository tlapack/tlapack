/// @file test_hetrf.cpp Test the Bunch-Kaufman factorization of a symmetric
/// matrix

/// @author Hugh M Kadhem, University of California Berkeley, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "TestUploMatrix.hpp"

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lansy.hpp>

// Other routines
#include <tlapack/lapack/hetrf.hpp>

using namespace tlapack;

template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR ipiv_t>
void ldlmul(const uplo_t& uplo,
            const matrix_t& L,
            const ipiv_t& ipiv,
            matrix_t& E,
            const bool hermitian);
template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR ipiv_t>
void ldlmul(const uplo_t& uplo,
            const matrix_t& L,
            const ipiv_t& ipiv,
            matrix_t&& E,
            const bool hermitian);

#define TESTUPLO_TYPES_TO_TEST                                          \
    (TestUploMatrix<float, size_t, Uplo::Lower, Layout::ColMajor>),     \
        (TestUploMatrix<float, size_t, Uplo::Upper, Layout::ColMajor>), \
        (TestUploMatrix<float, size_t, Uplo::Lower, Layout::RowMajor>), \
        (TestUploMatrix<float, size_t, Uplo::Upper, Layout::RowMajor>)

TEMPLATE_TEST_CASE("Bunch-Kaufman factorization of a symmetric matrix",
                   "[hetrf]",
                   TLAPACK_TYPES_TO_TEST,
                   TESTUPLO_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using vector_t = vector_type<TestType>;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;
    Create<vector_t> new_vector;

    // MatrixMarket reader
    MatrixMarket mm;

    using variant_t = pair<HetrfVariant, idx_t>;
    const variant_t variant = GENERATE((variant_t(HetrfVariant::Blocked, 2)),
                                       (variant_t(HetrfVariant::Blocked, 3)),
                                       (variant_t(HetrfVariant::Blocked, 7)),
                                       (variant_t(HetrfVariant::Blocked, 10)));
    const idx_t n = GENERATE(10, 19, 30);
    const Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);
    const Op invariant = GENERATE(Op::Trans, Op::ConjTrans);

    DYNAMIC_SECTION("n = " << n << " uplo = " << uplo << " variant = "
                           << (char)variant.first << " nb = " << variant.second
                           << " invariant = " << (char)invariant)
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        // Create matrices
        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);
        std::vector<T> L_;
        auto L = new_matrix(L_, n, n);
        std::vector<T> E_;
        auto E = new_matrix(E_, n, n);
        std::vector<int> ipiv_;
        auto ipiv = new_vector(ipiv_, n);

        // Update A with random numbers
        mm.random(uplo, A);
        if (invariant == Op::ConjTrans) {
            // Hermitian matrix so diagonal should be real.
            for (int i = 0; i < n; ++i)
                A(i, i) = real(A(i, i));
        }

        lacpy(uplo, A, L);
        real_t normA = tlapack::lansy(tlapack::MAX_NORM, uplo, A);

        // Run the LDL factorization
        HetrfOpts opts;
        opts.variant = variant.first;
        opts.nb = variant.second;
        opts.invariant = invariant;
        int info = hetrf(uplo, L, ipiv, opts);

        // Check that the factorization was successful
        REQUIRE(info == 0);

        // Initialize E with L * D * L**T
        ldlmul(uplo, L, ipiv, E, (invariant == Op::ConjTrans));

        // Check that the factorization is correct
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (((uplo == Uplo::Upper) && (i <= j)) ||
                    ((uplo == Uplo::Lower) && (i >= j)))
                    E(i, j) -= A(i, j);
            }
        }

        // Check for relative error: norm(A-cholesky(A))/norm(A)
        real_t error = tlapack::lansy(tlapack::MAX_NORM, uplo, E) / normA;
        CHECK(error <= tol);
    }
}

// This helper function multiplies together the LDL factors stored in L and
// writes them to E.
template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR ipiv_t>
void ldlmul(const uplo_t& uplo,
            const matrix_t& L,
            const ipiv_t& ipiv,
            matrix_t& E,
            const bool hermitian)
{
    using T = type_t<matrix_t>;
    const auto n = ncols(L);
    const bool upper = uplo == Uplo::Upper;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            E(i, j) = T(0);
    // The upper/lower unitriangular factor is stored as a sequence of
    // transformations P(0)L(0)...P(i)L(i)...P(n-2)L(n-2). Each L(i) has s_i
    // non-identity columns, where s_i is 1 or 2. Each P(i) is a transposition.
    // The columns of L(i) are stored in the super/sub-diagonal columns of L,
    // at position i from the end/beginning.
    // The transposition P(i) is stored in ipiv.
    // We compute the product by multiplying each standard basis vector on the
    // right by the sequence of factors.
    for (int j = 0; j < n; ++j) {
        // j is the index of the output column being computed.
        // We start by setting it to the standard basis vector e_j.
        E(j, j) = T(1);
        // We multiply it through the factors L(i)^op P(i) in reverse.
        for (int i_ = 0; i_ < n; ++i_) {
            // i_ is the sequence index, and i is the index to the first
            // column/row of L(i_).
            int i = upper ? n - 1 - i_ : i_;
            // s and piv are the indices to the columns that will be swapped by
            // P(i_).
            int s = i;
            int piv = ipiv[i];
            if (piv < 0) {
                piv = -piv - 1;
                s += upper ? -1 : 1;
                ++i_;
            }
            if (s != piv) std::swap(E(s, j), E(piv, j));
            if (upper)
                for (; i >= s; --i)
                    for (int k = 0; k < s; ++k)
                        E(i, j) +=
                            (hermitian ? conj(L(k, i)) : L(k, i)) * E(k, j);
            else
                for (; i <= s; ++i)
                    for (int k = s + 1; k < n; ++k)
                        E(i, j) +=
                            (hermitian ? conj(L(k, i)) : L(k, i)) * E(k, j);
        }
        // Multiply the column through the diagonal factor D.
        for (int i = 0; i < n; ++i) {
            int piv = ipiv[i];
            if (piv >= 0) {
                // D(i,i) is a 1x1 diagonal block stored at L(i,i).
                E(i, j) *= L(i, i);
            }
            else {
                // D(i,i) is a 2x2 symmetric/hermitian block stored in the
                // trailing 2x2 upper/lower triangle block at L(i,i).
                T D12, D21;
                if (upper) {
                    D12 = L(i, i + 1);
                    D21 = hermitian ? conj(D12) : D12;
                }
                else {
                    D21 = L(i + 1, i);
                    D12 = hermitian ? conj(D21) : D21;
                }
                T x1 = L(i, i) * E(i, j) + D12 * E(i + 1, j);
                T x2 = D21 * E(i, j) + L(i + 1, i + 1) * E(i + 1, j);
                E(i, j) = x1;
                E(i + 1, j) = x2;
                ++i;
            }
        }
        // Multiply the column through the sequence of factors P(i)L(i), in the
        // reverse order now.
        for (int i_ = n - 1; i_ >= 0; --i_) {
            int i = upper ? n - 1 - i_ : i_;
            int s = i;
            int piv = ipiv[i];
            if (piv < 0) {
                piv = -piv - 1;
                i += upper ? 1 : -1;
                --i_;
            }
            if (upper)
                for (; i >= s; --i)
                    for (int k = 0; k < s; ++k)
                        E(k, j) += L(k, i) * E(i, j);
            else
                for (; i <= s; ++i)
                    for (int k = s + 1; k < n; ++k)
                        E(k, j) += L(k, i) * E(i, j);
            std::swap(E(s, j), E(piv, j));
        }
    }
}

// Overload to accept r-value E, needed explicitly because it is not a const
// reference.
template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR ipiv_t>
void ldlmul(const uplo_t& uplo,
            const matrix_t& L,
            const ipiv_t& ipiv,
            matrix_t&& E,
            const bool hermitian)
{
    return ldlmul(uplo, L, ipiv, E, hermitian);
}
