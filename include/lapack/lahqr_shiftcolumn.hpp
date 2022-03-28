/// @file lahqr_shiftcolumn.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlaqr1.f
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LAHQR_SHIFTCOLUMN_HH__
#define __LAHQR_SHIFTCOLUMN_HH__

#include <complex>

#include "lapack/utils.hpp"
#include "lapack/types.hpp"

namespace lapack
{

   template <
       class matrix_t,
       class vector_t,
       typename TH = type_t<matrix_t>,
       typename idx_t = size_type<matrix_t>,
       typename real_t = real_type<TH>,
       enable_if_t<!is_complex<TH>::value, bool> = true>
   int lahqr_shiftcolumn(matrix_t &H, vector_t &v, std::complex<real_t> s1, std::complex<real_t> s2)
   {

      // Using
      using blas::imag;
      using blas::real;
      using blas::abs;

      // Constants
      idx_t n = ncols(H);
      const real_t zero(0);

      // Check arguments
      lapack_error_if((n != 2 and n != 3), -1);
      lapack_error_if(n != nrows(H), -1);
      lapack_error_if((idx_t)size(v) != n, -2);

      if (n == 2)
      {
         auto s = abs(H(0, 0) - real(s2)) + abs(imag(s2)) + abs(H(1, 0));
         if (s == zero)
         {
            v[0] = zero;
            v[1] = zero;
         }
         else
         {
            auto h21s = H(1, 0) / s;
            v[0] = h21s * H(0, 1) + (H(0, 0) - real(s1)) * ((H(0, 0) - real(s2)) / s) - imag(s1) * (imag(s2) / s);
            v[1] = h21s * (H(0, 0) + H(1, 1) - real(s1) - real(s2));
         }
      }
      else
      {
         auto s = abs(H(0, 0) - real(s2)) + abs(imag(s2)) + abs(H(1, 0)) + abs(H(2, 0));
         if (s == zero)
         {
            v[0] = zero;
            v[1] = zero;
            v[2] = zero;
         }
         else
         {
            auto h21s = H(1, 0) / s;
            auto h31s = H(2, 0) / s;
            v[0] = (H(0, 0) - real(s1)) * ((H(0, 0) - real(s2)) / s) - imag(s1) * (imag(s2) / s) + H(0, 1) * h21s + H(0, 2) * h31s;
            v[1] = h21s * (H(0, 0) + H(1, 1) - real(s1) - real(s2)) + H(1, 2) * h31s;
            v[2] = h31s * (H(0, 0) + H(2, 2) - real(s1) - real(s2)) + h21s * H(2, 1);
         }
      }
      return 0;
   }

   template <
       class matrix_t,
       class vector_t,
       typename TH = type_t<matrix_t>,
       typename idx_t = size_type<matrix_t>,
       typename real_t = real_type<TH>,
       enable_if_t<is_complex<TH>::value, bool> = true>
   int lahqr_shiftcolumn(matrix_t &H, vector_t &v, std::complex<real_t> s1, std::complex<real_t> s2)
   {

      using blas::abs1;

      // Constants
      idx_t n = ncols(H);
      const real_t rzero(0);
      const TH zero(0);

      // Check arguments
      lapack_error_if((n != 2 and n != 3), -1);
      lapack_error_if(n != nrows(H), -1);
      lapack_error_if((idx_t)size(v) != n, -2);

      //TODO: fix index 1 -> 0
      if (n == 2)
      {
         auto s = abs1(H(0, 0) - s2) + abs1(H(1, 0));
         if (s == rzero)
         {
            v[0] = zero;
            v[1] = zero;
         }
         else
         {
            auto h21s = H(1, 0) / s;
            v[0] = h21s * H(0, 1) + (H(0, 0) - s1) * ((H(0, 0) - s2) / s);
            v[1] = h21s * (H(0, 0) + H(1, 1) - s1 - s2);
         }
      }
      else
      {
         auto s = abs1(H(0, 0) - s2) + abs1(H(1, 0)) + abs1(H(2, 0));
         if (s == rzero)
         {
            v[0] = zero;
            v[1] = zero;
            v[2] = zero;
         }
         else
         {
            auto h21s = H(1, 0) / s;
            auto h31s = H(2, 0) / s;
            v[0] = (H(0, 0) - s1) * ((H(0, 0) - s2) / s) + H(0, 1) * h21s + H(0, 2) * h31s;
            v[1] = h21s * (H(0, 0) + H(1, 1) - s1 - s2) + H(1, 2) * h31s;
            v[2] = h31s * (H(0, 0) + H(2, 2) - s1 - s2) + h21s * H(2, 1);
         }
      }
      return 0;
   }

} // lapack

#endif // __LAHQR_SHIFTCOLUMN_HH__
