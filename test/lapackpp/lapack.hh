// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_HH
#define LAPACK_HH

#include "lapack/defines.h"
#include "lapack/util.hh"
#include "tlapack/legacy_api/lapack.hpp"

namespace lapack {

using tlapack::ladiv;
using tlapack::lapy2;
using tlapack::lapy3;

using tlapack::legacy::lacpy;
using tlapack::legacy::lange;
using tlapack::legacy::lanhe;
using tlapack::legacy::lansy;
using tlapack::legacy::lantr;
using tlapack::legacy::larf;
using tlapack::legacy::larfb;
using tlapack::legacy::larfg;
using tlapack::legacy::larft;
using tlapack::legacy::larnv;
using tlapack::legacy::lascl;
using tlapack::legacy::laset;
using tlapack::legacy::lassq;

using tlapack::legacy::geqr2;
using tlapack::legacy::potrf;
using tlapack::legacy::potrs;
using tlapack::legacy::ung2r;
using tlapack::legacy::unm2r;
using tlapack::legacy::unmqr;

}  // namespace lapack

#endif  // LAPACK_HH
