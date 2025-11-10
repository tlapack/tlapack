/// @file test_trevc_backsolve.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test eigenvector calculations.
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>

// Other routines
#include <tlapack/blas/gemv.hpp>
#include <tlapack/lapack/trevc_backsolve.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("TREVC protect div correctly protects against overflow",
                   "[eigenvectors][trevc]",
                   float,
                   double)
{
    using T = TestType;

    T sf_max = safe_max<T>();
    T sf_min = safe_min<T>();

    SECTION("very simple, obviously no scaling needed")
    {
        T a = T(5);
        T b = T(1);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }

    SECTION("no scaling needed (case 1 in paper)")
    {
        T a = T(0.4);
        T b = sf_min * T(0.5);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }

    SECTION("scaling needed (case 2 in paper)")
    {
        T a = T(0.6);
        T b = sf_min * T(0.5);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale < T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }

    SECTION("no scaling needed (case 3 in paper)")
    {
        T a = sf_max;
        T b = T(2);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }

    SECTION("no scaling needed (but close) (case 4 in paper)")
    {
        T a = sf_max / T(2);
        T b = T(0.6);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }

    SECTION("scaling needed (case 5 in paper)")
    {
        T a = sf_max / T(1.1);
        T b = T(0.6);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale < T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }
}

TEMPLATE_TEST_CASE(
    "TREVC protect div correctly protects against overflow for complex cases",
    "[eigenvectors][trevc]",
    std::complex<float>,
    std::complex<double>)
{
    using T = TestType;
    using real_t = real_type<T>;

    real_t sf_max = safe_max<real_t>();
    real_t sf_min = safe_min<real_t>();

    SECTION("Very simple case, no scaling needed")
    {
        T a = T(5);
        T b = T(1);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }

    SECTION("No scaling needed (case 1 in paper)")
    {
        T a = T(0.2, 0.3);
        T b = sf_min * T(0.5, 0.6);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }

    SECTION("Scaling needed (case 2 in paper)")
    {
        T a = T(0.3, 0.4);
        T b = sf_min * T(0.5, 0.6);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale < real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }

    SECTION("no scaling needed (case 3 in paper)")
    {
        T a = sf_max;
        T b = T(2);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }

    SECTION("no scaling needed (but close) (case 4 in paper)")
    {
        T a = T(sf_max / real_t(4), sf_max / real_t(5));
        T b = T(0.6, 1.2);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }

    SECTION("scaling needed (case 5 in paper)")
    {
        T a = sf_max / T(1.1);
        T b = T(0.6);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale < real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
    }
}