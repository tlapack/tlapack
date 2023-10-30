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
    static_assert(Arithmetic<int>);
    static_assert(Arithmetic<int&>);
    static_assert(Arithmetic<const int&> == false);
    static_assert(Arithmetic<int&&>);
    static_assert(Arithmetic<const int&&> == false);
}

TEST_CASE("Concept Vector works as expected", "[concept]")
{
    static_assert(Vector<std::vector<float>>);
}

TEST_CASE("Concept Complex works as expected", "[concept]")
{
    static_assert(Complex<std::complex<float>>);
    static_assert(Complex<tlapack::NaNPropagComplex<float>>);
}

TEST_CASE("Concept SliceableMatrix works as expected", "[concept]")
{
    #ifdef TLAPACK_TEST_EIGEN
    using matrix_t = Eigen::Matrix<std::complex<float>, -1, -1, 1, -1, -1>;
    static_assert(SliceableMatrix<matrix_t>);

    matrix_t A(2, 2);
    auto B =
        tlapack::slice(A, std::pair<int, int>{0, 1}, std::pair<int, int>{0, 1});
    static_assert(Matrix<decltype(B)>);

    B(0, 0);
    auto m = tlapack::nrows(B);
    auto n = tlapack::ncols(B);
    auto s = tlapack::size(B);
    static_assert(std::integral<decltype(m)>);
    static_assert(std::integral<decltype(n)>);
    static_assert(std::integral<decltype(s)>);
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

    static_assert(Workspace<matrix_t>);
    static_assert(Workspace<vector_t>);

    // Functor
    tlapack::Create<matrix_t> new_matrix;
    tlapack::Create<vector_t> new_vector;

    {
        std::vector<T> A_;
        auto A = new_matrix(A_, 3, 4);

        // Reshapes into matrices and vectors
        auto [B, w] = tlapack::reshape(A, 1, 1);
        static_assert(Workspace<decltype(w)>);
        auto [c, w2] = tlapack::reshape(w, 1);
        static_assert(Workspace<decltype(w2)>);

        // Raise an exception if the size is larger than the original
        CHECK_THROWS(tlapack::reshape(A, 4, 4));
        CHECK_THROWS(tlapack::reshape(A, 15, 1));
        CHECK_THROWS(tlapack::reshape(A, 15));

        // Reshapes contiguous data
        {
            auto [B, w0] = tlapack::reshape(A, 4, 3);
            static_assert(Workspace<decltype(w0)>);
            auto [C, w1] = tlapack::reshape(A, 3, 3);
            static_assert(Workspace<decltype(w1)>);
            auto [D, w2] = tlapack::reshape(A, 2, 4);
            static_assert(Workspace<decltype(w2)>);
            auto [E, w3] = tlapack::reshape(A, 10);
            static_assert(Workspace<decltype(w3)>);
        }

        // Reshapes non-contiguous matrix
        {
            auto A0 = tlapack::slice(A, std::pair{0, 2}, std::pair{0, 3});
            auto [B, w0] = tlapack::reshape(A0, 2, 2);
            static_assert(Workspace<decltype(w0)>);
            auto [C, w1] = tlapack::reshape(A0, 1, 3);
            static_assert(Workspace<decltype(w1)>);
            auto [d, w2] = tlapack::reshape(A0, 2);
            static_assert(Workspace<decltype(w2)>);
            auto [e, w3] = tlapack::reshape(A0, 3);
            static_assert(Workspace<decltype(w3)>);
        }

        // Reshapes non-contiguous vector
        {
            auto A0 = tlapack::slice(A, std::pair{0, 3}, 0);
            auto A1 = tlapack::slice(A, 0, std::pair{0, 4});
            try {
                auto [B, w0] = tlapack::reshape(A0, 3, 1);
                static_assert(Workspace<decltype(w0)>);
                auto [C, w1] = tlapack::reshape(A0, 2, 1);
                static_assert(Workspace<decltype(w1)>);
                auto [D, w2] = tlapack::reshape(A1, 1, 4);
                static_assert(Workspace<decltype(w2)>);
                auto [E, w3] = tlapack::reshape(A1, 1, 2);
                static_assert(Workspace<decltype(w3)>);
            }
            catch (...) {
                auto [B, w0] = tlapack::reshape(A0, 1, 3);
                static_assert(Workspace<decltype(w0)>);
                auto [C, w1] = tlapack::reshape(A0, 1, 2);
                static_assert(Workspace<decltype(w1)>);
                auto [D, w2] = tlapack::reshape(A1, 4, 1);
                static_assert(Workspace<decltype(w2)>);
                auto [E, w3] = tlapack::reshape(A1, 2, 1);
                static_assert(Workspace<decltype(w3)>);
            }
            auto [f, w4] = tlapack::reshape(A0, 2);
            static_assert(Workspace<decltype(w4)>);
            auto [g, w5] = tlapack::reshape(A1, 3);
            static_assert(Workspace<decltype(w5)>);
        }
    }

    {
        std::vector<T> v_;
        auto v = new_vector(v_, 10);

        // Reshapes into matrices and vectors
        auto [B, w] = tlapack::reshape(v, 1, 1);
        static_assert(Workspace<decltype(w)>);
        auto [c, w2] = tlapack::reshape(w, 1);
        static_assert(Workspace<decltype(w2)>);

        // Raise an exception if the size is larger than the original
        CHECK_THROWS(tlapack::reshape(v, 11));
        CHECK_THROWS(tlapack::reshape(v, 5, 3));
        CHECK_THROWS(tlapack::reshape(v, 1, 12));

        // Reshapes contiguous data successfully
        {
            auto [b, w0] = tlapack::reshape(v, 4, 2);
            static_assert(Workspace<decltype(w0)>);
            auto [c, w1] = tlapack::reshape(v, 8, 1);
            static_assert(Workspace<decltype(w1)>);
            auto [d, w2] = tlapack::reshape(v, 1, 8);
            static_assert(Workspace<decltype(w2)>);
            auto [e, w3] = tlapack::reshape(v, 7);
            static_assert(Workspace<decltype(w3)>);
        }
    }
}

TEMPLATE_TEST_CASE(
    "Concepts ConstructableMatrix and ConstructableVector work as expected",
    "[concept]",
    TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using vector_t = tlapack::vector_type<matrix_t>;
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;

    #define M 5
    #define N 2
    #define K 7

    static_assert(ConstructableMatrix<matrix_t>);
    static_assert(ConstructableVector<vector_t>);

    // Functor
    tlapack::Create<matrix_t> new_matrix;
    tlapack::Create<vector_t> new_vector;
    tlapack::CreateStatic<matrix_t, M, N> new_MbyN_matrix;
    tlapack::CreateStatic<vector_t, K> new_K_vector;

    std::vector<T> A_;
    auto A = new_matrix(A_, 3, 4);
    REQUIRE(tlapack::nrows(A) == 3);
    REQUIRE(tlapack::ncols(A) == 4);

    T B_[M * N];
    auto B = new_MbyN_matrix(B_);
    REQUIRE(tlapack::nrows(B) == M);
    REQUIRE(tlapack::ncols(B) == N);

    std::vector<T> v_;
    auto v = new_vector(v_, 10);
    REQUIRE(tlapack::size(v) == 10);

    T w_[K];
    auto w = new_K_vector(w_);
    REQUIRE(tlapack::size(w) == K);
}

#endif
