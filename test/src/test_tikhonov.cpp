/// @file test_tikhonov.cpp Test if tikhonov regularized least squares problem
/// is successfully solved
/// @author L. Carlos Gutierrez, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "testutils.hpp"
//
#include <tlapack/lapack/tik_bidiag_elden.hpp>
#include <tlapack/lapack/tik_qr.hpp>
#include <tlapack/lapack/tik_svd.hpp>
#include <tlapack/lapack/tkhnv.hpp>
#include <tlapack/plugins/stdvector.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Testing all cases of Tikhonov",
                   "[tikhonov check]",
                   TLAPACK_TYPES_TO_TEST)
//    (tlapack::LegacyMatrix<std::complex<float>,
//   std::size_t,
//   tlapack::Layout::ColMajor>))
// (tlapack::LegacyMatrix<double, std::size_t, tlapack::Layout::ColMajor>))
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    Create<matrix_t> new_matrix;

    const idx_t m = GENERATE(1, 2, 12, 20, 30);
    const idx_t n = GENERATE(1, 2, 3, 7, 8);
    const idx_t k = GENERATE(1, 7, 12, 19);
    const real_t lambda = real_t(GENERATE(1e-6, 7.5, 1e6));
    // const real_t lambda = real_t(GENERATE(1e-2, 7.5, 56, 1e2));
    // const real_t lambda = real_t(GENERATE(7.5));
    // const real_t lambda = real_t(GENERATE(1e-2));
    // idx_t m = 1, n = 1, k = 1;
    // real_t lambda = 1e-2;

    //
    using variant_t = TikVariant;
    const variant_t variant =
        // GENERATE((variant_t(TikVariant::QR)), (variant_t(TikVariant::Elden)),
        //          (variant_t(TikVariant::SVD)));
        // GENERATE((variant_t(TikVariant::Elden)),
        // (variant_t(TikVariant::SVD)));
        GENERATE((variant_t(TikVariant::QR)));
    DYNAMIC_SECTION(" m = " << m << " n = " << n << " k = " << k << " lambda = "
                            << lambda << " variant = " << (char)variant)

    {
        std::cout.precision(5);
        std::cout << std::scientific << std::showpos;
        if (m >= n) {
            // eps is the machine precision, and tol is the tolerance we accept
            // for tests to pass

            const real_t eps = ulp<real_t>();
            const real_t tol = 2 * real_t(max(m, k)) * eps;

            // Declare matrices
            std::vector<T> A_;
            auto A = new_matrix(A_, m, n);
            std::vector<T> A_copy_;
            auto A_copy = new_matrix(A_copy_, m, n);
            std::vector<T> b_;
            auto b = new_matrix(b_, m, k);
            std::vector<T> bcopy_;
            auto bcopy = new_matrix(bcopy_, m, k);
            std::vector<T> x_;
            auto x = new_matrix(x_, n, k);
            std::vector<T> y_;
            auto y = new_matrix(y_, n, k);

            MatrixMarket mm;
            mm.random(A);
            mm.random(b);

            real_t normA = lange(FROB_NORM, A);

            if ((variant != TikVariant::QR) || (lambda < normA)) {
                lacpy(GENERAL, b, bcopy);
                lacpy(GENERAL, A, A_copy);

                TikOpts opts;
                opts.variant = variant;
                tkhnv(A, b, lambda, opts);

                // Check routine

                lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), x);

                // Compute b - A *x -> b
                gemm(NO_TRANS, NO_TRANS, real_t(-1), A_copy, x, real_t(1),
                     bcopy);

                // Compute A.H*(b - A x) -> y
                gemm(CONJ_TRANS, NO_TRANS, real_t(1), A_copy, bcopy, y);

                // Compute A.H*(b - A x) - (lambda^2)*x -> y
                for (idx_t j = 0; j < k; j++)
                    for (idx_t i = 0; i < n; i++)
                        y(i, j) -= (lambda) * (lambda)*x(i, j);

                real_t normr = lange(FROB_NORM, y);

                real_t normb = lange(FROB_NORM, bcopy);

                real_t normx = lange(FROB_NORM, x);

                if (normr > tol * (normA * (normb + normA * normx) +
                                   abs(lambda) * abs(lambda) * normx)) {
                    std::cout << "\nnormr = " << normr;

                    std::cout << "\ntol * scalar= "
                              << tol * (normA * (normb + normA * normx) +
                                        abs(lambda) * abs(lambda) * normx)
                              << "\n";
                }

                // Note: tikqr is unstable when lambda > ||A||. Therefore, the
                // check will not be applied

                // introduced in Catch2 2.8.0
                Catch::StringMaker<real_t>::precision = 15;
                CHECK(normr <= tol * (normA * (normb + normA * normx) +
                                      abs(lambda) * abs(lambda) * normx));
            }
        }
    }
}