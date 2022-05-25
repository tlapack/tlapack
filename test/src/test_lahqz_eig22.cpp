/// @file test_lahqz_eig22.cpp
/// @brief Test eigenvalues of generalized eigenvalues
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("check that lahqz_eig22 gives correct eigenvalues", "[generalizedeigenvalues]", real_types_to_test)
{
   srand(1);

   using matrix_t = TestType;
   using T = type_t<matrix_t>;
   using idx_t = size_type<matrix_t>;
   using real_t = real_type<T>;

   const real_type<T> eps = ulp<real_type<T>>();

   // auto matrix_type = GENERATE(as<std::string>{}, "Random", "Near overflow");
   auto matrix_type = GENERATE(as<std::string>{}, "Near overflow");

   // Define the matrices and vectors
   std::unique_ptr<T[]> A_(new T[4]);
   std::unique_ptr<T[]> B_(new T[4]);
   std::unique_ptr<std::complex<real_t>[]> As_(new std::complex<real_t>[4]);

   // This only works for legacy matrix, we really work on that construct_matrix function
   auto A = legacyMatrix<T, layout<matrix_t>>(2, 2, &A_[0], 2);
   auto B = legacyMatrix<T, layout<matrix_t>>(2, 2, &B_[0], 2);
   auto As = legacyMatrix<std::complex<real_t>, layout<matrix_t>>(2, 2, &As_[0], 2);

   if (matrix_type == "Random")
   {
      // Generate a random matrix in A
      for (idx_t j = 0; j < 2; ++j)
         for (idx_t i = 0; i < 2; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      // Generate a random matrix in B
      for (idx_t j = 0; j < 2; ++j)
         for (idx_t i = 0; i < 2; ++i)
            B(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
   }
   if (matrix_type == "Near overflow")
   {
      const real_t large_num = safe_max<real_t>() * ulp<real_t>();

      for (idx_t j = 0; j < 2; ++j)
         for (idx_t i = 0; i < 2; ++i)
            A(i, j) = large_num;

      for (idx_t j = 0; j < 2; ++j)
         for (idx_t i = 0; i < 2; ++i)
            B(i, j) = large_num;
   }

   B(1, 0) = T(0);

   T beta1, beta2;
   std::complex<real_t> alpha1, alpha2;
   lahqz_eig22(A, B, alpha1, alpha2, beta1, beta2);

   auto normA = lange(max_norm, A);
   auto normB = lange(max_norm, B);

   // Check first eigenvalue
   {
      for (idx_t i = 0; i < 2; ++i)
         for (idx_t j = 0; j < 2; ++j)
            As(i, j) = beta1 * A(i, j) - alpha1 * B(i, j);

      auto normAs = lange(max_norm, As);
      auto As_scale = real_t(0) / sqrt(normAs);
      auto det = (As_scale * As(0, 0)) * (As_scale * As(1, 1)) - 
                 (As_scale * As(1, 0)) * (As_scale * As(0, 1));
      auto temp = max(abs(beta1) * normA, abs(alpha1) * normB);
      CHECK(abs(det) <= 1.0e1 * eps * temp);
   }

   // Check second eigenvalue
   {
      for (idx_t i = 0; i < 2; ++i)
         for (idx_t j = 0; j < 2; ++j)
            As(i, j) = beta2 * A(i, j) - alpha2 * B(i, j);

      auto normAs = lange(max_norm, As);
      auto det = As(0, 0) * As(1, 1) - As(1, 0) * As(0, 1);
      auto temp = max(abs(beta2) * normA, abs(alpha2) * normAs);
      CHECK(abs(det) <= 1.0e1 * eps * temp);
   }
}
