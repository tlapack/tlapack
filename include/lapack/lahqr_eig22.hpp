/// @file lahqr_eig22.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlaqr1.f
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LAHQR_EIG22_HH__
#define __LAHQR_EIG22_HH__

#include <complex>

#include "lapack/utils.hpp"
#include "lapack/types.hpp"

namespace lapack
{

   template <
       typename T,
       typename real_t = real_type<T>
   >
   void lahqr_eig22(T a11, T a12, T a21, T a22, std::complex<real_t>& s1, std::complex<real_t>& s2)
   {

      // Using
      using blas::abs1;

      // Constants
      const real_t rzero(0);
      const real_t two(2);
      const T zero(0);

      auto s = abs1( a11 ) + abs1( a12 ) + abs1( a21 ) + abs1( a22 );
      if( s == rzero ) {
         s1 = zero;
         s2 = zero;
         return;
      }

      // TODO: check this calculation
      a11 = a11/s;
      a12 = a12/s;
      a21 = a21/s;
      a22 = a22/s;
      auto tr = (a11+a22);
      std::complex<real_t> det = (a11-a22)*(a11-a22) + 4.0*a12*a21;
      auto rtdisc = sqrt(det);
      // This root formula is known to be unstable due to cancellation,
      // it might make more sense to change it
      s1 = s*(tr + rtdisc)/two;
      s2 = s*(tr - rtdisc)/two;

   }

} // lapack

#endif // __LAHQR_EIG22_HH__
