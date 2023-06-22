// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_HH
#define BLAS_HH

#include "blas/defines.h"
#include "blas/util.hh"
#include "tlapack/legacy_api/blas.hpp"

namespace blas {

// using std::min;
// using std::max;

// using tlapack::real_type;
// using tlapack::complex_type;
// using tlapack::scalar_type;
// using tlapack::is_complex;

using tlapack::Diag;
using tlapack::Layout;
using tlapack::Op;
using tlapack::Side;
using tlapack::Uplo;

// =============================================================================
// Level 1 BLAS template implementations

using tlapack::legacy::asum;
using tlapack::legacy::axpy;
using tlapack::legacy::copy;
using tlapack::legacy::dot;
using tlapack::legacy::dotu;
using tlapack::legacy::iamax;
using tlapack::legacy::nrm2;
using tlapack::legacy::rot;
using tlapack::legacy::rotg;
using tlapack::legacy::rotm;
using tlapack::legacy::rotmg;
using tlapack::legacy::scal;
using tlapack::legacy::swap;

// =============================================================================
// Level 2 BLAS template implementations

using tlapack::legacy::gemv;
using tlapack::legacy::ger;
using tlapack::legacy::geru;
using tlapack::legacy::hemv;
using tlapack::legacy::her;
using tlapack::legacy::her2;
using tlapack::legacy::symv;
using tlapack::legacy::syr;
using tlapack::legacy::syr2;
using tlapack::legacy::trmv;
using tlapack::legacy::trsv;

// =============================================================================
// Level 3 BLAS template implementations

using tlapack::legacy::gemm;
using tlapack::legacy::hemm;
using tlapack::legacy::her2k;
using tlapack::legacy::herk;
using tlapack::legacy::symm;
using tlapack::legacy::syr2k;
using tlapack::legacy::syrk;
using tlapack::legacy::trmm;
using tlapack::legacy::trsm;

}  // namespace blas

#endif  // BLAS_HH
