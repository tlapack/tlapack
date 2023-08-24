/// @file test/src/test_mdspanplugin.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @brief Tests for the mdspan plugin.
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tlapack/base/utils.hpp>

#undef I  // I is defined in complex.h and generates a conflict with templates
// in Catch2

#include <tlapack/plugins/mdspan.hpp>

TEST_CASE("STD layouts work as expected", "[plugins]")
{
    using std::experimental::dextents;
    using std::experimental::layout_left;
    using std::experimental::layout_right;
    using std::experimental::layout_stride;
    using std::experimental::mdspan;

    using tlapack::col;
    using tlapack::cols;
    using tlapack::Layout;
    using tlapack::layout;
    using tlapack::ncols;
    using tlapack::nrows;
    using tlapack::row;
    using tlapack::rows;
    using tlapack::slice;

    using idx_t = std::size_t;
    using range = std::pair<idx_t, idx_t>;

    using my_dextents = dextents<idx_t, 2>;

    mdspan<float, my_dextents, layout_left> A(nullptr, 11, 13);
    mdspan<float, my_dextents, layout_right> B(nullptr, 7, 5);
    mdspan<float, my_dextents, layout_stride> C(
        nullptr, layout_stride::mapping<my_dextents>(
                     my_dextents{3, 4}, std::array<idx_t, 2>{1, 3}));

    SECTION("mdspan layouts are correctly translated")
    {
        CHECK(layout<decltype(A)> == Layout::ColMajor);
        CHECK(layout<decltype(B)> == Layout::RowMajor);
        CHECK(layout<decltype(C)> == Layout::Unspecified);
    }

    SECTION("Slicing a mdspan sometimes turns the layout into Unspecified")
    {
        CHECK(layout<decltype(slice(A, range{0, 1}, range{0, 1}))> ==
              Layout::Unspecified);
        CHECK(layout<decltype(slice(B, range{0, 1}, range{0, 1}))> ==
              Layout::Unspecified);
        CHECK(layout<decltype(slice(C, range{0, 1}, range{0, 1}))> ==
              Layout::Unspecified);

        SECTION("layout_left (Column-major contiguous data)")
        {
            CHECK(layout<decltype(slice(A, range{0, nrows(A)}, range{0, 1}))> ==
                  Layout::Unspecified);
            CHECK(layout<decltype(slice(A, range{0, nrows(A)}, 1))> ==
                  Layout::Strided);
            CHECK(layout<decltype(slice(A, range{0, 1}, range{0, ncols(A)}))> ==
                  Layout::Unspecified);
            CHECK(layout<decltype(slice(A, 1, range{0, ncols(A)}))> ==
                  Layout::Strided);

            CHECK(layout<decltype(cols(A, range{0, 1}))> == Layout::ColMajor);
            CHECK(layout<decltype(col(A, 1))> == Layout::Strided);
            CHECK(layout<decltype(rows(A, range{0, 1}))> ==
                  Layout::Unspecified);
            CHECK(layout<decltype(row(A, 1))> == Layout::Strided);
        }
    }
}
