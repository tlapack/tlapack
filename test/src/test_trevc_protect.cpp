/// @file test_trevc_protect.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test overflow protection routines for TREVC.
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
#include <tlapack/lapack/trevc_protect.hpp>

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
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("no scaling needed (case 1 in paper)")
    {
        T a = T(0.4);
        T b = sf_min * T(0.5);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("scaling needed (case 2 in paper)")
    {
        T a = T(0.6);
        T b = sf_min * T(0.5);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale < T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("no scaling needed (case 3 in paper)")
    {
        T a = sf_max;
        T b = T(2);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("no scaling needed (but close) (case 4 in paper)")
    {
        T a = sf_max / T(2);
        T b = T(0.6);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("scaling needed (case 5 in paper)")
    {
        T a = sf_max / T(1.1);
        T b = T(0.6);
        T scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale < T(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
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
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("No scaling needed (case 1 in paper)")
    {
        T a = T(0.2, 0.3);
        T b = sf_min * T(0.5, 0.6);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("Scaling needed (case 2 in paper)")
    {
        T a = T(0.3, 0.4);
        T b = sf_min * T(0.5, 0.6);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale < real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("no scaling needed (case 3 in paper)")
    {
        T a = sf_max;
        T b = T(2);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("no scaling needed (but close) (case 4 in paper)")
    {
        T a = T(sf_max / real_t(4), sf_max / real_t(5));
        T b = T(0.6, 1.2);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale == real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("scaling needed (case 5 in paper)")
    {
        T a = sf_max / T(1.1);
        T b = T(0.6);
        real_t scale = trevc_protectdiv(a, b, sf_min, sf_max);
        REQUIRE(scale < real_t(1));
        T c = (scale * a) / b;
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }
}

TEMPLATE_TEST_CASE("TREVC protect update correctly protects against overflow",
                   "[eigenvectors][trevc]",
                   float,
                   double)
{
    using T = TestType;

    T sf_max = safe_max<T>();
    T sf_min = safe_min<T>();

    SECTION("very simple, obviously no scaling needed")
    {
        T y = T(5);
        T t = T(-1);
        T x = T(2);

        T scale = trevc_protectupdate(abs(y), abs(t), abs(x), sf_max);

        REQUIRE(scale == T(1));
        T c = (scale * y) - t * (scale * x);
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("no scaling needed (case 1 in paper)")
    {
        T y = -sf_max * T(0.4);
        T t = sf_max;
        T x = T(0.5);

        T scale = trevc_protectupdate(abs(y), abs(t), abs(x), sf_max);

        REQUIRE(scale == T(1));
        T c = (scale * y) - t * (scale * x);
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("scaling needed (case 2 in paper)")
    {
        T y = -sf_max * T(0.6);
        T t = sf_max;
        T x = T(0.5);

        T scale = trevc_protectupdate(abs(y), abs(t), abs(x), sf_max);

        REQUIRE(scale < T(1));
        T c = (scale * y) - t * (scale * x);
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("no scaling needed (case 3 in paper)")
    {
        T y = sf_max * T(0.4);
        T t = sf_max * T(0.01);
        T x = T(20.0);

        T scale = trevc_protectupdate(abs(y), abs(t), abs(x), sf_max);

        REQUIRE(scale == T(1));
        T c = (scale * y) - t * (scale * x);
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("scaling needed (case 4 in paper)")
    {
        T y = sf_max * T(0.4);
        T t = sf_max * T(0.1);
        T x = T(20.0);

        T scale = trevc_protectupdate(abs(y), abs(t), abs(x), sf_max);

        REQUIRE(scale < T(1));
        T c = (scale * y) - t * (scale * x);
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }
}

TEMPLATE_TEST_CASE(
    "TREVC protect update correctly protects against overflow for complex "
    "cases",
    "[eigenvectors][trevc]",
    std::complex<float>,
    std::complex<double>)
{
    using T = TestType;
    using real_t = real_type<T>;

    real_t sf_max = safe_max<real_t>();
    real_t sf_min = safe_min<real_t>();

    SECTION("very simple, obviously no scaling needed")
    {
        T y = T(5, 6);
        T t = T(-1, 3);
        T x = T(2, 4);

        T scale = trevc_protectupdate(abs1(y), abs1(t), abs1(x), sf_max);

        REQUIRE(scale == T(1));
        T c = (scale * y) - t * (scale * x);
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("no scaling needed (case 1 in paper)")
    {
        T y = sf_max * T(0.5);
        T t = -sf_max * T(0.25, 0.25);
        T x = T(0.25, 0.25);

        real_t scale = trevc_protectupdate(abs1(y), abs1(t), abs1(x), sf_max);

        REQUIRE(scale == real_t(1));
        T c = (scale * y) - t * (scale * x);
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("scaling needed (case 2 in paper)")
    {
        T y = sf_max * T(0.9);
        T t = -sf_max * T(0.9, 0.9);
        T x = T(0.45, 0.45);

        real_t scale = trevc_protectupdate(abs1(y), abs1(t), abs1(x), sf_max);

        REQUIRE(scale == real_t(0.5));
        T c = (scale * y) - t * (scale * x);
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("no scaling needed (case 3 in paper)")
    {
        T y = sf_max *
              T(0.4, 0.4);  // abs1(y) = sf_max * 0.8, abs(y) = sf_max * 0.565
        T t = sf_max * T(0.01);  // abs(t) = abs1(t) = sf_max * 0.01
        T x = T(1.1);            // abs(x) = abs1(x) = 1

        real_t scale = trevc_protectupdate(abs1(y), abs1(t), abs1(x), sf_max);

        REQUIRE(scale == real_t(1));
        T c = (scale * y) - t * (scale * x);
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }

    SECTION("scaling needed (case 4 in paper)")
    {
        T y = sf_max *
              T(0.4, 0.4);  // abs1(y) = sf_max * 0.8, abs(y) = sf_max * 0.565
        T t = sf_max * T(0.5);  // abs(t) = abs1(t) = sf_max * 0.1
        T x = T(1.1);           // abs(x) = abs1(x) = 1

        real_t scale = trevc_protectupdate(abs1(y), abs1(t), abs1(x), sf_max);

        REQUIRE(scale < real_t(1));
        T c = (scale * y) - t * (scale * x);
        REQUIRE(!(isnan(c) || isinf(c)));
        REQUIRE(abs1(c) <= sf_max);
    }
}
