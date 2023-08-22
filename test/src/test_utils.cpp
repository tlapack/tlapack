/// @file test_utils.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @brief Test utils
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

using namespace tlapack;

TEST_CASE("Random generator is consistent if seed is fixed", "[utils]")
{
    rand_generator gen;
    gen.seed(6845315);

    CHECK(gen() == 1225581775);
    CHECK(gen() == 1985311242);
    CHECK(gen() == 300629471);
    CHECK(gen() == 2636314308);
    CHECK(gen() == 1603395911);
    CHECK(gen() == 393807335);
    CHECK(gen() == 3641191292);
}

TEMPLATE_TEST_CASE("is_matrix works", "[utils]", TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using namespace tlapack::traits::internal;

    CHECK(is_matrix<matrix_t>);
}

TEST_CASE("is_matrix and is_vector work", "[utils]")
{
    using namespace tlapack::traits::internal;

    CHECK(!is_matrix<std::vector<float> >);
    CHECK(!is_matrix<LegacyVector<float> >);

    CHECK(is_vector<std::vector<float> >);
    CHECK(is_vector<LegacyVector<float> >);

    CHECK(!is_matrix<float>);
    CHECK(!is_matrix<std::complex<double> >);

    CHECK(!is_vector<float>);
    CHECK(!is_vector<std::complex<double> >);
}
