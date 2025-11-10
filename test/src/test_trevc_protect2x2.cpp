/// @file test_trevc_protect2x2.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test overflow protection routines for TREVC (solving 2x2 system).
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

TEMPLATE_TEST_CASE(
    "TREVC protect 2x2 correctly solves 2x2 systems that don't need scaling",
    "[eigenvectors][trevc]",
    float,
    double)
{
    using T = TestType;

    T sf_max = safe_max<T>();
    T sf_min = safe_min<T>();

    T tol = ulp<T>() * 10;

    T r1 = T(4);
    T r2 = T(5);
    T x1 = r1;
    T x2 = r2;
    T scale = T(1);

    // 0 -> a is largest
    // 1 -> b is largest
    // 2 -> c is largest
    // 3 -> d is largest
    int ilargest = GENERATE(0, 1, 2, 3);

    T a, b, c, d;
    if (ilargest == 0) {
        a = T(5);
        b = T(1);
        c = T(2);
        d = T(3);
    }
    else if (ilargest == 1) {
        a = T(1);
        b = T(5);
        c = T(2);
        d = T(3);
    }
    else if (ilargest == 2) {
        a = T(2);
        b = T(1);
        c = T(5);
        d = T(3);
    }
    else {
        a = T(2);
        b = T(1);
        c = T(3);
        d = T(5);
    }

    DYNAMIC_SECTION(" ilargest = " << ilargest)
    {
        trevc_2x2solve(a, b, c, d, x1, x2, scale, sf_min, sf_max);

        REQUIRE(scale == T(1));
        REQUIRE(!(isnan(x1) || isinf(x1)));
        REQUIRE(!(isnan(x2) || isinf(x2)));
        REQUIRE(abs1(x1) <= sf_max);
        REQUIRE(abs1(x2) <= sf_max);

        // Check residual
        T res1 = a * x1 + b * x2 - r1;
        T res2 = c * x1 + d * x2 - r2;

        T norma = std::max({abs1(a), abs1(b), abs1(c), abs1(d)});
        T normr = std::max(abs1(r1), abs1(r2));

        REQUIRE(abs1(res1) <= tol * (norma + normr));
        REQUIRE(abs1(res2) <= tol * (norma + normr));
    }
}

TEMPLATE_TEST_CASE(
    "TREVC protect 2x2 correctly solves 2x2 systems that need scaling",
    "[eigenvectors][trevc]",
    float,
    double)
{
    using T = TestType;

    T sf_max = safe_max<T>();
    T sf_min = safe_min<T>();

    T tol = ulp<T>() * 10;

    T a(0.001);
    T b(0.001);
    T c(0.001);
    T d(0.002);

    T r1 = sf_max;
    T r2 = sf_max / T(4);
    T x1 = r1;
    T x2 = r2;

    T scale;
    trevc_2x2solve(a, b, c, d, x1, x2, scale, sf_min, sf_max);

    REQUIRE(scale <= T(1));
    REQUIRE(!(isnan(x1) || isinf(x1)));
    REQUIRE(!(isnan(x2) || isinf(x2)));
    REQUIRE(abs1(x1) <= sf_max);
    REQUIRE(abs1(x2) <= sf_max);

    // Check residual
    T res1 = a * x1 + b * x2 - scale * r1;
    T res2 = c * x1 + d * x2 - scale * r2;
    T norma = std::max({abs1(a), abs1(b), abs1(c), abs1(d)});
    T normr = std::max(abs1(r1), abs1(r2));
    REQUIRE(abs1(res1) <= tol * (norma + normr));
    REQUIRE(abs1(res2) <= tol * (norma + normr));
}

TEMPLATE_TEST_CASE(
    "TREVC protect 2x2 correctly solves complex 2x2 systems that don't need "
    "scaling",
    "[eigenvectors][trevc]",
    std::complex<float>,
    std::complex<double>)
{
    using T = TestType;
    using real_t = real_type<T>;

    real_t sf_max = safe_max<real_t>();
    real_t sf_min = safe_min<real_t>();

    real_t tol = ulp<real_t>() * 10;

    real_t r1r = real_t(4);
    real_t r1i = real_t(0);
    real_t r2r = real_t(5);
    real_t r2i = real_t(0);
    real_t x1r = r1r;
    real_t x1i = r1i;
    real_t x2r = r2r;
    real_t x2i = r2i;
    real_t scale = real_t(1);

    // 0 -> a is largest
    // 1 -> b is largest
    // 2 -> c is largest
    // 3 -> d is largest
    int ilargest = GENERATE(0, 1, 2, 3);

    real_t ar, ai, br, bi, cr, ci, dr, di;
    if (ilargest == 0) {
        ar = real_t(5);
        ai = real_t(0);
        br = real_t(1);
        bi = real_t(1);
        cr = real_t(2);
        ci = real_t(0);
        dr = real_t(3);
        di = real_t(1);
    }
    else if (ilargest == 1) {
        ar = real_t(1);
        ai = real_t(1);
        br = real_t(5);
        bi = real_t(0);
        cr = real_t(2);
        ci = real_t(0);
        dr = real_t(3);
        di = real_t(1);
    }
    else if (ilargest == 2) {
        ar = real_t(2);
        ai = real_t(0);
        br = real_t(1);
        bi = real_t(1);
        cr = real_t(5);
        ci = real_t(0);
        dr = real_t(3);
        di = real_t(1);
    }
    else {
        ar = real_t(2);
        ai = real_t(0);
        br = real_t(1);
        bi = real_t(1);
        cr = real_t(3);
        ci = real_t(1);
        dr = real_t(5);
        di = real_t(0);
    }

    DYNAMIC_SECTION(" ilargest = " << ilargest)
    {
        trevc_2x2solve(ar, ai, br, bi, cr, ci, dr, di, x1r, x1i, x2r, x2i,
                       scale, sf_min, sf_max);

        T x1(x1r, x1i);
        T x2(x2r, x2i);
        T a(ar, ai);
        T b(br, bi);
        T c(cr, ci);
        T d(dr, di);
        T r1(r1r, r1i);
        T r2(r2r, r2i);

        REQUIRE(scale == T(1));
        REQUIRE(!(isnan(x1) || isinf(x1)));
        REQUIRE(!(isnan(x2) || isinf(x2)));
        REQUIRE(abs1(x1) <= sf_max);
        REQUIRE(abs1(x2) <= sf_max);

        // Check residual
        T res1 = a * x1 + b * x2 - r1;
        T res2 = c * x1 + d * x2 - r2;

        real_t norma = std::max({abs1(a), abs1(b), abs1(c), abs1(d)});
        real_t normr = scale * std::max(abs1(r1), abs1(r2));

        REQUIRE(abs1(res1) <= tol * (norma + normr));
        REQUIRE(abs1(res2) <= tol * (norma + normr));
    }
}

TEMPLATE_TEST_CASE(
    "TREVC protect 2x2 correctly solves complex 2x2 systems that do need "
    "scaling",
    "[eigenvectors][trevc]",
    std::complex<float>,
    std::complex<double>)
{
    using T = TestType;
    using real_t = real_type<T>;

    real_t sf_max = safe_max<real_t>();
    real_t sf_min = safe_min<real_t>();

    real_t tol = ulp<real_t>() * 10;

    real_t r1r = sf_max * real_t(0.5);
    real_t r1i = real_t(0);
    real_t r2r = sf_max * real_t(0.5);
    real_t r2i = real_t(0);
    real_t x1r = r1r;
    real_t x1i = r1i;
    real_t x2r = r2r;
    real_t x2i = r2i;
    real_t scale;

    real_t ar = real_t(0.001);
    real_t ai = real_t(0);
    real_t br = real_t(0.001);
    real_t bi = real_t(1);
    real_t cr = real_t(0.001);
    real_t ci = real_t(0);
    real_t dr = real_t(0.002);
    real_t di = real_t(1);

    trevc_2x2solve(ar, ai, br, bi, cr, ci, dr, di, x1r, x1i, x2r, x2i, scale,
                   sf_min, sf_max);

    T x1(x1r, x1i);
    T x2(x2r, x2i);
    T a(ar, ai);
    T b(br, bi);
    T c(cr, ci);
    T d(dr, di);
    T r1(r1r, r1i);
    T r2(r2r, r2i);

    REQUIRE(scale < real_t(1));
    REQUIRE(!(isnan(x1) || isinf(x1)));
    REQUIRE(!(isnan(x2) || isinf(x2)));
    REQUIRE(abs1(x1) <= sf_max);
    REQUIRE(abs1(x2) <= sf_max);

    // Check residual
    T res1 = a * x1 + b * x2 - (scale * r1);
    T res2 = c * x1 + d * x2 - (scale * r2);

    real_t norma = std::max({abs1(a), abs1(b), abs1(c), abs1(d)});
    real_t normr = scale * std::max(abs1(r1), abs1(r2));

    REQUIRE(abs1(res1) <= tol * (norma + normr));
    REQUIRE(abs1(res2) <= tol * (norma + normr));
}
