/// @file test_rscl.cpp Test the reciprocal scaling routine.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// std::complex wrapper that propagates NaNs
#include "NaNPropagComplex.hpp"

// Auxiliary routines
#include <tlapack/base/constants.hpp>
#include <tlapack/blas/axpy.hpp>
#include <tlapack/blas/copy.hpp>
#include <tlapack/blas/iamax.hpp>
#include <tlapack/blas/nrm2.hpp>
#include <tlapack/lapack/ladiv.hpp>

// Other routines
#include <tlapack/lapack/rscl.hpp>

using namespace tlapack;

#ifdef TLAPACK_TEST_EIGEN
    #define TEST_TYPES_RSCL float, double, Eigen::half
#else
    #define TEST_TYPES_RSCL float, double
#endif

TEMPLATE_TEST_CASE("reciprocal scaling works on limit cases",
                   "[rscl]",
                   TEST_TYPES_RSCL)
{
    using real_t = TestType;
    {
        std::vector<real_t> v = {real_t(1), real_t(2), real_t(3), real_t(4),
                                 real_t(5)};
        rscl(real_t(1), v);
        CHECK(v == std::vector<real_t>({real_t(1), real_t(2), real_t(3),
                                        real_t(4), real_t(5)}));
    }
    {
        std::vector<real_t> v = {real_t(1), real_t(2), real_t(3), real_t(4),
                                 real_t(5)};
        rscl(real_t(2), v);
        CHECK(v == std::vector<real_t>({real_t(0.5), real_t(1), real_t(1.5),
                                        real_t(2), real_t(2.5)}));
    }
    {
        std::vector<real_t> v = {real_t(1), real_t(2), real_t(3), real_t(4),
                                 real_t(5)};
        rscl(real_t(0.5), v);
        CHECK(v == std::vector<real_t>({real_t(2), real_t(4), real_t(6),
                                        real_t(8), real_t(10)}));
    }
    {
        using T = NaNPropagComplex<real_t>;
        using trustR = long double;
        using trustT = NaNPropagComplex<trustR>;

        // scaling constants
        const real_t Inf = std::numeric_limits<real_t>::infinity();
        const real_t NaN = std::numeric_limits<real_t>::quiet_NaN();
        const real_t M = std::numeric_limits<real_t>::max();
        const real_t m = std::numeric_limits<real_t>::min();
        const real_t denormMin = std::numeric_limits<real_t>::denorm_min();
        const real_t safeMax = safe_max<real_t>();
        const real_t safeMin = safe_min<real_t>();
        const real_t eps = ulp<real_t>();

        // constants
        const real_t half = real_t(0.5);
        const real_t one = real_t(1);
        const real_t two = real_t(2);
        const real_t four = real_t(4);
        const real_t zero = real_t(0);
        const real_t thirty = real_t(30);
        const real_t oneOverThirty = real_t(1) / real_t(30);

        // Theoretical limits for the relative error
        // See Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
        // Section 3.6.
        const real_t u = uroundoff<real_t>();
        const real_t gamma7 = (real_t(7) * u) / (one - real_t(7) * u);
        const real_t tol = sqrt(two) * gamma7;

        const std::vector<T> alpha_vec = {
            // Close to Max
            T(M, eps), T(half * M, half * M), T(eps, M),
            // Close to safeMax
            T(safeMax, four * eps), T(safeMax, safeMax * (four * eps)),
            T(safeMax, safeMax), T(safeMax * (four * eps), safeMax),
            T(four * eps, safeMax),
            // Close to safeMax/2
            T(half * safeMax, four * eps),
            T(half * safeMax, two * safeMax * eps),
            T(half * safeMax, half * safeMax),
            T(half * safeMax, half * safeMax * (one + four * eps)),
            T(half * safeMax * (one + four * eps), half * safeMax),
            T(two * safeMax * eps, half * safeMax),
            T(four * eps, half * safeMax),
            // Close to 1
            T(one, four * eps), T(one, one), T(four * eps, one),
            // Close to 30
            T(thirty, four * eps), T(thirty, thirty), T(four * eps, thirty),
            // Close to 1/30
            T(oneOverThirty, four * eps), T(oneOverThirty, oneOverThirty),
            T(four * eps, oneOverThirty),
            // Close to eps
            T(eps, four * eps * eps), T(eps, eps), T(four * eps * eps, eps),
            // Close to safeMin
            T(safeMin, safeMin * four * eps), T(safeMin, safeMin),
            T(safeMin * four * eps, safeMin),
            // Close to safeMin/2
            T(half * safeMin, (two * safeMin) * eps),
            T(half * safeMin, half * safeMin),
            T((two * safeMin) * eps, half * safeMin),
            // Close to min
            T(m, m * four * eps), T(m, m), T(m * four * eps, m),
            // Close to min/2
            T(half * m, (two * m) * eps), T(half * m, half * m),
            T((two * m) * eps, half * m),
            // Close to denormMin
            T(denormMin, zero), T(denormMin, denormMin), T(zero, denormMin),
            // Inf
            T(Inf, Inf), T(Inf, zero), T(Inf, one), T(zero, Inf), T(one, Inf),
            T(NaN, NaN), T(NaN, zero), T(NaN, one), T(zero, NaN), T(one, NaN),
            T(Inf, NaN), T(NaN, Inf)};

        for (T alpha : alpha_vec) {
            for (size_t k = 0; k < 2; ++k) {
                const trustT alpha_ = trustT(real(alpha), imag(alpha));
                const std::vector<T> v = alpha_vec;

                const std::vector<T> v_ref = [v, alpha_] {
                    std::vector<T> v_ref(v.size());
                    for (size_t i = 0; i < v.size(); ++i) {
                        const trustT vi_ =
                            ladiv(trustT(real(v[i]), imag(v[i])), alpha_);
                        v_ref[i] = T((real_t)real(vi_), (real_t)imag(vi_));
                    }
                    return v_ref;
                }();

                const std::vector<T> v_naive = [v, alpha] {
                    std::vector<T> v_naive(v.size());
                    for (size_t i = 0; i < v.size(); ++i) {
                        v_naive[i] = v[i] / alpha;
                    }
                    return v_naive;
                }();

                std::vector<T> v_scal = v;
                rscl(alpha, v_scal);

                INFO("safemax = " << std::scientific << safeMax);
                INFO("u = " << std::scientific << u);
                INFO("tol = " << std::scientific << tol);

                INFO("alpha = " << std::scientific << alpha);
                INFO("a = " << std::scientific
                            << real(alpha) +
                                   imag(alpha) * (imag(alpha) / real(alpha)));
                INFO("b = " << std::scientific
                            << imag(alpha) +
                                   real(alpha) * (real(alpha) / imag(alpha)));

                size_t i = 0;
                for (; i < v.size(); ++i) {
                    if (alpha == zero) {
                        // If alpha is zero and v[i] is zero, then v_scal[i]
                        // must be NaN.
                        if (!isnan(v_scal[i])) break;
                    }
                    else if (isnan(alpha) || isnan(v[i])) {
                        // If alpha or v[i] are NaN, then v_scal[i] must be NaN.
                        if (!isnan(v_scal[i])) break;
                    }
                    else if (isnan(v_scal[i]) && !isnan(v_ref[i])) {
                        // If v_scal[i] is NaN and v_ref[i] isn't, then
                        // (1/alpha) * v[i] must be a NaN. Moreover, v_ref[i]
                        // must be an Inf.
                        const T invAlpha = [one, alpha_]() {
                            trustT aux = ladiv(trustT(one), alpha_);
                            return T((real_t)real(aux), (real_t)imag(aux));
                        }();
                        if (!(isinf(v_ref[i]) && isnan(invAlpha * v[i]))) break;
                    }
                    else if (isinf(v_scal[i]) && !isinf(v_ref[i])) {
                        // If v_scal[i] is Inf and v_ref[i] isn't, then v_ref[i]
                        // must be a NaN. This is an exceptional case where
                        // either the real or imaginary part of alpha is zero
                        // and v[i] is infinite.
                        if (!(isnan(v_ref[i]) &&
                              (real(alpha) == zero || imag(alpha) == zero) &&
                              isinf(v[i])))
                            break;
                    }

                    if (isnan(v_ref[i]) || isinf(v_ref[i])) {
                        // If v_ref[i] is NaN or Inf, then v_scal[i] must
                        // be a NaN or Inf as well.
                        if (!isinf(v_scal[i]) && !isnan(v_scal[i])) break;
                    }
                    else {
                        const real_t rel_bnd = tol * tlapack::abs(v_ref[i]);
                        const real_t err = tlapack::abs(v_scal[i] - v_ref[i]);
                        const real_t err_naive =
                            tlapack::abs(v_naive[i] - v_ref[i]);

                        if (rel_bnd > zero || err_naive > zero) {
                            // If either the relative bound or err_naive is
                            // greater than zero, we use both of them to
                            // determine if the error is acceptable.
                            if (rel_bnd == zero) {
                                if (err > err_naive) break;
                            }
                            else if (err_naive == zero) {
                                if (err > rel_bnd) break;
                            }
                            else {
                                if (err > rel_bnd && err > err_naive) break;
                            }
                        }
                        else {
                            // If both tol and err_naive are zero, then we can
                            // only use the absolute error to determine if the
                            // error is acceptable.
                            if (err >= m) break;
                        }
                    }
                }

                if (i != v.size()) {
                    const real_t rel_bnd = tol * tlapack::abs(v_ref[i]);
                    const real_t err = tlapack::abs(v_scal[i] - v_ref[i]);
                    const real_t err_naive =
                        tlapack::abs(v_naive[i] - v_ref[i]);
                    UNSCOPED_INFO("v[" << i << "] = " << std::scientific
                                       << v[i]);
                    UNSCOPED_INFO("v[" << i << "]/alpha = " << std::scientific
                                       << v_scal[i] << " != " << v_ref[i]
                                       << " (ref)");
                    UNSCOPED_INFO("Error: " << std::scientific << err);
                    UNSCOPED_INFO("Error from naive approach: "
                                  << std::scientific << err_naive);
                    UNSCOPED_INFO("rel_bnd: " << std::scientific << rel_bnd);
                }

                CHECK(i == v.size());

                // Do another round with the conjugate of alpha.
                alpha = T(real(alpha), -imag(alpha));
            }
        }
    }
}
