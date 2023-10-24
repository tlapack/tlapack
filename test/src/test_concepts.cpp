/// @file test_concepts.cpp Test concepts in concepts.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Load NaNPropagComplex
#include "NaNPropagComplex.hpp"

#if __cplusplus >= 202002L

using namespace tlapack::concepts;

template <TLAPACK_COMPLEX T>
void f(T x)
{
    return;
}

TEST_CASE("Concept Arithmetic works as expected", "[concept]")
{
    REQUIRE(Arithmetic<int>);
    REQUIRE(Arithmetic<int&>);
    REQUIRE(Arithmetic<const int&> == false);
    REQUIRE(Arithmetic<int&&>);
    REQUIRE(Arithmetic<const int&&> == false);
}

TEST_CASE("Concept Vector works as expected", "[concept]")
{
    REQUIRE(Vector<std::vector<float>>);
}

TEST_CASE("Concept Complex works as expected", "[concept]")
{
    REQUIRE(Complex<std::complex<float>>);
    REQUIRE(Complex<tlapack::NaNPropagComplex<float>>);
}

TEST_CASE("Concept SliceableMatrix works as expected", "[concept]")
{
    #ifdef TLAPACK_TEST_EIGEN
    using matrix_t = Eigen::Matrix<std::complex<float>, -1, -1, 1, -1, -1>;
    REQUIRE(SliceableMatrix<matrix_t>);

    matrix_t A(2, 2);
    auto B =
        tlapack::slice(A, std::pair<int, int>{0, 1}, std::pair<int, int>{0, 1});
    REQUIRE(Matrix<decltype(B)>);

    B(0, 0);
    auto m = tlapack::nrows(B);
    auto n = tlapack::ncols(B);
    auto s = tlapack::size(B);
    REQUIRE(std::integral<decltype(m)>);
    REQUIRE(std::integral<decltype(n)>);
    REQUIRE(std::integral<decltype(s)>);
    #endif
}

TEMPLATE_TEST_CASE("Concept Workspace works as expected",
                   "[concept]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using vector_t = tlapack::vector_type<matrix_t>;
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;

    // Functor
    tlapack::Create<matrix_t> new_matrix;
    tlapack::Create<vector_t> new_vector;

    {
        std::vector<T> A_;
        auto A = new_matrix(A_, 3, 4);

        // Reshapes into matrices and vectors
        auto [B, w] = tlapack::reshape(A, 1, 1);
        REQUIRE(Workspace<decltype(w)>);
        auto [c, w2] = tlapack::reshape(w, 1);
        REQUIRE(Workspace<decltype(w2)>);

        // Raise an exception if the size is larger than the original
        CHECK_THROWS(tlapack::reshape(A, 4, 4));
        CHECK_THROWS(tlapack::reshape(A, 15, 1));
        CHECK_THROWS(tlapack::reshape(A, 15));

        // Reshapes contiguous data
        if constexpr (tlapack::legacy::is_legacy_type<matrix_t>) {
            const bool is_contiguous = (A.m * A.n <= 1) ||
                                       (A.layout == tlapack::Layout::ColMajor &&
                                        (A.ldim == A.m || A.n <= 1)) ||
                                       (A.layout == tlapack::Layout::RowMajor &&
                                        (A.ldim == A.n || A.m <= 1));
            if (is_contiguous) {
                auto [B, w0] = tlapack::reshape(A, 4, 3);
                REQUIRE(Workspace<decltype(w0)>);
                auto [C, w1] = tlapack::reshape(A, 3, 3);
                REQUIRE(Workspace<decltype(w1)>);
                auto [D, w2] = tlapack::reshape(A, 2, 4);
                REQUIRE(Workspace<decltype(w2)>);
                auto [E, w3] = tlapack::reshape(A, 10);
                REQUIRE(Workspace<decltype(w3)>);
            }
        }
    #ifdef TLAPACK_TEST_EIGEN
        else if constexpr (tlapack::eigen::is_eigen_type<matrix_t>) {
            const bool is_contiguous =
                (A.size() <= 1) ||
                (matrix_t::IsRowMajor
                     ? (A.outerStride() == A.cols() || A.rows() <= 1)
                     : (A.outerStride() == A.rows() || A.cols() <= 1));
            if (is_contiguous) {
                auto [B, w0] = tlapack::reshape(A, 4, 3);
                REQUIRE(Workspace<decltype(w0)>);
                auto [C, w1] = tlapack::reshape(A, 3, 3);
                REQUIRE(Workspace<decltype(w1)>);
                auto [D, w2] = tlapack::reshape(A, 2, 4);
                REQUIRE(Workspace<decltype(w2)>);
                auto [E, w3] = tlapack::reshape(A, 10);
                REQUIRE(Workspace<decltype(w3)>);
            }
        }
    #endif
    #ifdef TLAPACK_TEST_MDSPAN
        else if constexpr (tlapack::mdspan::is_mdspan_type<matrix_t>) {
            const bool is_contiguous =
                (A.size() <= 1) ||
                (A.stride(0) == 1 &&
                 (A.stride(1) == A.extent(0) || A.extent(1) <= 1)) ||
                (A.stride(1) == 1 &&
                 (A.stride(0) == A.extent(1) || A.extent(0) <= 1));
            if (is_contiguous) {
                auto [B, w0] = tlapack::reshape(A, 4, 3);
                REQUIRE(Workspace<decltype(w0)>);
                auto [C, w1] = tlapack::reshape(A, 3, 3);
                REQUIRE(Workspace<decltype(w1)>);
                auto [D, w2] = tlapack::reshape(A, 2, 4);
                REQUIRE(Workspace<decltype(w2)>);
                auto [E, w3] = tlapack::reshape(A, 10);
                REQUIRE(Workspace<decltype(w3)>);
            }
        }
    #endif
    }

    {
        std::vector<T> v_;
        auto v = new_vector(v_, 10);

        // Reshapes into matrices and vectors
        auto [B, w] = tlapack::reshape(v, 1, 1);
        REQUIRE(Workspace<decltype(w)>);
        auto [c, w2] = tlapack::reshape(w, 1);
        REQUIRE(Workspace<decltype(w2)>);

        // Raise an exception if the size is larger than the original
        CHECK_THROWS(tlapack::reshape(v, 11));
        CHECK_THROWS(tlapack::reshape(v, 5, 3));
        CHECK_THROWS(tlapack::reshape(v, 1, 12));

        // Reshapes contiguous data successfully
        if constexpr (tlapack::legacy::is_legacy_type<matrix_t>) {
            const bool is_contiguous = (v.inc == 1) || (v.n <= 1);
            if (is_contiguous) {
                auto [b, w0] = tlapack::reshape(v, 4, 2);
                REQUIRE(Workspace<decltype(w0)>);
                auto [c, w1] = tlapack::reshape(v, 8, 1);
                REQUIRE(Workspace<decltype(w1)>);
                auto [d, w2] = tlapack::reshape(v, 1, 8);
                REQUIRE(Workspace<decltype(w2)>);
                auto [e, w3] = tlapack::reshape(v, 7);
                REQUIRE(Workspace<decltype(w3)>);
            }
        }
    #ifdef TLAPACK_TEST_EIGEN
        else if constexpr (tlapack::eigen::is_eigen_type<matrix_t>) {
            const bool is_contiguous = (v.size() <= 1 || v.innerStride() == 1);
            if (is_contiguous) {
                auto [b, w0] = tlapack::reshape(v, 4, 2);
                REQUIRE(Workspace<decltype(w0)>);
                auto [c, w1] = tlapack::reshape(v, 8, 1);
                REQUIRE(Workspace<decltype(w1)>);
                auto [d, w2] = tlapack::reshape(v, 1, 8);
                REQUIRE(Workspace<decltype(w2)>);
                auto [e, w3] = tlapack::reshape(v, 7);
                REQUIRE(Workspace<decltype(w3)>);
            }
        }
    #endif
    #ifdef TLAPACK_TEST_MDSPAN
        else if constexpr (tlapack::mdspan::is_mdspan_type<matrix_t>) {
            const bool is_contiguous = (v.size() <= 1 || v.stride(0) == 1);
            if (is_contiguous) {
                auto [b, w0] = tlapack::reshape(v, 4, 2);
                REQUIRE(Workspace<decltype(w0)>);
                auto [c, w1] = tlapack::reshape(v, 8, 1);
                REQUIRE(Workspace<decltype(w1)>);
                auto [d, w2] = tlapack::reshape(v, 1, 8);
                REQUIRE(Workspace<decltype(w2)>);
                auto [e, w3] = tlapack::reshape(v, 7);
                REQUIRE(Workspace<decltype(w3)>);
            }
        }
    #endif
    }
}

#endif