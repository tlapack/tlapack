/// @file test_trmm_out.cpp
/// @author Ella Addison-Taylor, Kyle Cunningham, University of Colorado Denver,
/// USA
/// @brief Test banded, Hermitian matrix Cholesky factorization.
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


// Other routines
#include <iomanip>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/pbtrf.hpp>


using namespace tlapack;




TEMPLATE_TEST_CASE("triagular matrix-matrix multiplication is backward stable",
                  "[triangular matrix-matrix check]",
                  TLAPACK_TYPES_TO_TEST)
{
   using matrix_t = TestType;
   using T = type_t<matrix_t>;
   using idx_t = size_type<matrix_t>;
   typedef real_type<T> real_t;
   using range = pair<idx_t, idx_t>;


   // Functor
   Create<matrix_t> new_matrix;


   // MatrixMarket reader
   MatrixMarket mm;


   // const idx_t n = GENERATE(2, 13, 20, 25, 45, 131);
   const idx_t n = GENERATE(13);
   // const idx_t kd = GENERATE(1, 5, 7, 9, 12, 20);
   const idx_t kd = GENERATE(5);


   srand(3);


   // const Uplo uplo = GENERATE(Uplo::Upper, Uplo::Lower);
   const Uplo uplo = GENERATE(Uplo::Upper);


   // const size_t nb = GENERATE(1, 2, 3, 4, 7, 9, 11);
   const size_t nb = GENERATE(3);


   DYNAMIC_SECTION("n = " << n << " kd = " << kd << " Uplo = " << uplo
                          << " nb = " << nb)
   {
       if (nb <= kd && kd < n) {
           const real_t eps = ulp<real_t>();
           const real_t tol = real_t(n) * eps;


           std::vector<T> A_;
           auto A = new_matrix(A_, n, n);


           std::vector<T> A_orig_;
           auto A_orig = new_matrix(A_orig_, n, n);


           MatrixMarket mm;


           mm.random(A);


           for (idx_t i = 0; i < n; ++i) {
               A(i, i) += real_t(n);
           }


           BlockedAndBandedCholeskyOpts opts;
           opts.nb = nb;


           lacpy(Uplo::General, A, A_orig);


           pbtrf(uplo, A, kd, opts);


           if (uplo == Uplo::Upper) {
               real_t sum(0);
               for (idx_t j = 0; j < n; j++)
                   for (idx_t i = j + 1; i < n; i++)
                       sum += abs1(A(i, j) - A_orig(i, j));


               for (idx_t j = 0; j < n - kd; j++)
                   for (idx_t i = kd + 1 + j; i < n; ++i)
                       sum += abs1(A(j, i) - A_orig(j, i));


               CHECK(sum == real_t(0));


               auto temp2 = slice(A, range(0, n - kd), range(kd + 1, n));
               laset(uplo, T(0), T(0), temp2);


               for (idx_t j = 0; j < n; j++)
                   for (idx_t i = j + 1; i < n; i++)
                       A_orig(i, j) = T(0);


               for (idx_t j = 0; j < n - kd; j++)
                   for (idx_t i = kd + 1 + j; i < n; ++i)
                       A_orig(j, i) = T(0);


               mult_uhu(A);
           }
           else {
               real_t sum(0);
               for (idx_t j = 0; j < n; j++)
                   for (idx_t i = 0; i < j; i++)
                       sum += abs1(A(i, j) - A_orig(i, j));

               for (idx_t j = 0; j < n - kd; j++)
                   for (idx_t i = kd + 1 + j; i < n; ++i)
                       sum += abs1(A(i, j) - A_orig(i, j));

               CHECK(sum == real_t(0));
               auto temp2 = slice(A, range(kd + 1, n), range(0, n - kd));
               laset(uplo, T(0), T(0), temp2);

               for (idx_t j = 0; j < n; j++)
                   for (idx_t i = 0; i < j; i++)
                       A_orig(i, j) = T(0);

               for (idx_t j = 0; j < n - kd; j++)
                   for (idx_t i = kd + 1 + j; i < n; ++i)
                       A_orig(i, j) = T(0);
               mult_llh(A);
           }

           real_t normAbefore = lanhe(Norm::Fro, uplo, A_orig);

           for (idx_t i = 0; i < n; ++i) {
               for (idx_t j = 0; j < n; ++j) {
                   A(i, j) -= A_orig(i, j);
               }
           }


           real_t normA = lanhe(Norm::Fro, uplo, A);


           CHECK(normA <= tol * normAbefore);
       }
   }
}

