/// @file test_eigenplugin.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @brief Test plugin for integration with Eigen.
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <tlapack/base/utils.hpp>
#include <tlapack/plugins/eigen.hpp>

template <class block_t>
void test_block()
{
    CHECK(tlapack::eigen::internal::is_eigen_dense<block_t> == true);
    CHECK(tlapack::eigen::internal::is_eigen_matrix<block_t> == true);
    CHECK(tlapack::eigen::internal::is_eigen_block<block_t> == true);
}

template <class matrix_t>
void test_matrix()
{
    CHECK(tlapack::eigen::internal::is_eigen_dense<matrix_t> == true);
    CHECK(tlapack::eigen::internal::is_eigen_matrix<matrix_t> == true);
    CHECK(tlapack::eigen::internal::is_eigen_block<matrix_t> == false);
}

template <class map_t>
void test_map()
{
    CHECK(tlapack::eigen::internal::is_eigen_dense<map_t> == true);
    CHECK(tlapack::eigen::internal::is_eigen_matrix<map_t> == true);
    CHECK(tlapack::eigen::internal::is_eigen_block<map_t> == false);
}

template <class matrix_t>
void test_blocks()
{
    test_block<Eigen::Block<matrix_t>>();
    test_block<Eigen::Block<matrix_t, 3, 2>>();
    test_block<Eigen::Block<matrix_t, 3, 1>>();
    test_block<Eigen::Block<matrix_t, 1, 3>>();
    test_block<Eigen::Block<matrix_t, 1, -1>>();
    test_block<Eigen::Block<matrix_t, -1, 1>>();
}

template <class vector_t>
void test_vectorblocks()
{
    test_block<Eigen::VectorBlock<vector_t>>();
    test_block<Eigen::VectorBlock<vector_t, 1>>();
    test_block<Eigen::VectorBlock<vector_t, 0>>();
}

template <class matrix_t>
void test_maps()
{
    test_map<Eigen::Map<matrix_t>>();
    test_map<Eigen::Map<matrix_t, 0, Eigen::OuterStride<>>>();
    test_map<Eigen::Map<matrix_t, 0, Eigen::OuterStride<-1>>>();
    test_map<Eigen::Map<matrix_t, 0, Eigen::OuterStride<5>>>();
}

TEST_CASE("is_eigen_dense, is_eigen_block and is_eigen_map work", "[plugins]")
{
    CHECK(tlapack::eigen::internal::is_eigen_dense<int> == false);
    CHECK(tlapack::eigen::internal::is_eigen_dense<float> == false);
    CHECK(tlapack::eigen::internal::is_eigen_dense<std::complex<float>> ==
          false);
    CHECK(tlapack::eigen::internal::is_eigen_dense<std::string> == false);

    using M0 = Eigen::MatrixXd;
    using M1 = Eigen::VectorXd;
    using M2 = Eigen::Matrix<float, 2, 5>;
    using M3 = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;

    test_matrix<M0>();
    test_matrix<M1>();
    test_matrix<M2>();
    test_matrix<M3>();

    test_blocks<M0>();
    test_vectorblocks<M1>();
    test_blocks<M2>();
    test_blocks<M3>();

    test_maps<M0>();
    test_maps<M1>();
    test_maps<M2>();
    test_maps<M3>();

    test_block<Eigen::Block<Eigen::Block<M0>>>();
    test_block<Eigen::Block<Eigen::Map<M0>>>();
    test_map<Eigen::Map<Eigen::Block<M0>>>();
    test_map<Eigen::Map<Eigen::Map<M0>>>();

    using B = Eigen::VectorBlock<
        const Eigen::Block<Eigen::Block<Eigen::MatrixXd, -1, 1, true>, -1, 1,
                           false>,
        -1>;
    test_block<B>();
}

TEST_CASE("legacy_matrix works", "[plugins]")
{
    {
        Eigen::Matrix2d A;
        A << 1, 3, 2, 4;
        Eigen::MatrixXd A2 = A.block<1, 2>(1, 0);

        auto B = tlapack::legacy_matrix(A);
        auto C = tlapack::legacy::Matrix<double, Eigen::Index>{
            tlapack::layout<Eigen::Matrix2d>, A.rows(), A.cols(), A.data(),
            A.rows()};
        CHECK(B.layout == C.layout);
        CHECK(B.m == C.m);
        CHECK(B.n == C.n);
        CHECK(B.ptr == C.ptr);
        CHECK(B.ldim == C.ldim);

        auto B2 = tlapack::legacy_matrix(A2);
        auto C2 = tlapack::legacy::Matrix<double, Eigen::Index>{
            tlapack::layout<Eigen::MatrixXd>, A2.rows(), A2.cols(), A2.data(),
            A2.outerStride()};
        CHECK(B2.layout == C2.layout);
        CHECK(B2.m == C2.m);
        CHECK(B2.n == C2.n);
        CHECK(B2.ptr == C2.ptr);
        CHECK(B2.ldim == C2.ldim);

        auto B3 = tlapack::legacy_vector(A2);
        auto C3 = tlapack::legacy::Vector<double, Eigen::Index>{
            A2.size(), A2.data(), A2.innerStride()};
        CHECK(B3.n == C3.n);
        CHECK(B3.ptr == C3.ptr);
        CHECK(B3.inc == C3.inc);
    }
    {
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor> A(3, 5);
        A << 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2, -3, -4, -5;

        auto B = tlapack::legacy_matrix(A);
        auto C = tlapack::legacy::Matrix<float, Eigen::Index>{
            tlapack::layout<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>,
            A.rows(), A.cols(), A.data(), A.outerStride()};
        CHECK(B.layout == C.layout);
        CHECK(B.m == C.m);
        CHECK(B.n == C.n);
        CHECK(B.ptr == C.ptr);
        CHECK(B.ldim == C.ldim);

        Eigen::Matrix<float, -1, 1> A2 = A.col(3);
        auto B2 = tlapack::legacy_vector(A2);
        auto C2 = tlapack::legacy::Vector<float, Eigen::Index>{
            A2.size(), A2.data(), A2.innerStride()};
        CHECK(B2.n == C2.n);
        CHECK(B2.ptr == C2.ptr);
        CHECK(B2.inc == C2.inc);
    }
}

TEST_CASE("slice works", "[plugins]")
{
    using range = std::pair<Eigen::Index, Eigen::Index>;

    {
        Eigen::Matrix2d A;
        A << 1, 3, 2, 4;

        auto B = tlapack::slice(A, range{1, 2}, range{1, 2});
        CHECK(B(0, 0) == A(1, 1));
    }
    {
        Eigen::MatrixXd A(2, 2);
        A << 1, 3, 2, 4;

        auto B = tlapack::slice(A, range{1, 2}, range{1, 2});
        CHECK(B(0, 0) == A(1, 1));
    }
}

TEST_CASE("Layout works", "[plugins]")
{
    CHECK(tlapack::layout<Eigen::Matrix<float, -1, -1, 0, -1, -1>> ==
          tlapack::Layout::ColMajor);
    CHECK(tlapack::layout<Eigen::Matrix<float, -1, -1, 1, -1, -1>> ==
          tlapack::Layout::RowMajor);

    CHECK(tlapack::layout<Eigen::Block<Eigen::MatrixXd>> ==
          tlapack::Layout::ColMajor);

    CHECK(tlapack::layout<
              Eigen::Block<Eigen::Matrix<float, -1, -1, 1, -1, -1>>> ==
          tlapack::Layout::RowMajor);

    CHECK(tlapack::layout<
              Eigen::Block<Eigen::Matrix<float, -1, -1, 1, -1, -1>, -1, 1>> ==
          tlapack::Layout::Strided);
    CHECK(tlapack::layout<
              Eigen::Block<Eigen::Matrix<float, -1, -1, 1, -1, -1>, 1, -1>> ==
          tlapack::Layout::Strided);
    CHECK(tlapack::layout<
              Eigen::Block<Eigen::Matrix<float, -1, -1, 0, -1, -1>, -1, 1>> ==
          tlapack::Layout::Strided);
    CHECK(tlapack::layout<
              Eigen::Block<Eigen::Matrix<float, -1, -1, 0, -1, -1>, 1, -1>> ==
          tlapack::Layout::Strided);

    {
        using Derived =
            Eigen::Block<Eigen::Matrix<float, -1, -1, 1, -1, -1>, -1, 1, false>;
        CHECK(tlapack::layout<Derived> == tlapack::Layout::Strided);

        using Derived2 = Eigen::Map<Eigen::Matrix<float, -1, -1, 1, -1, -1>>;
        CHECK(tlapack::layout<Derived2> == tlapack::Layout::RowMajor);

        using Derived3 = Eigen::VectorBlock<Derived>;
        CHECK(tlapack::layout<Derived3> == tlapack::Layout::Strided);
    }
    {
        using B = Eigen::VectorBlock<
            Eigen::Block<Eigen::Block<Eigen::MatrixXd, -1, 1, true>, -1, 1,
                         false>,
            -1>;
        CHECK(tlapack::layout<B> == tlapack::Layout::Strided);
    }
}
