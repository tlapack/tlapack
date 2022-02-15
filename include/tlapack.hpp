// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_HH__
#define __TLAPACK_HH__

// BLAS

#include "tblas.hpp"

// =============================================================================
// Template LAPACK

// Auxiliary routines
// ------------------

#include "lapack/larf.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larft.hpp"
#include "lapack/larfb.hpp"
#include "lapack/lapy2.hpp"
#include "lapack/lapy3.hpp"
#include "lapack/ladiv.hpp"
#include "lapack/laset.hpp"
#include "lapack/lacpy.hpp"
#include "lapack/lange.hpp"
#include "lapack/lanhe.hpp"
#include "lapack/lansy.hpp"
#include "lapack/larnv.hpp"
#include "lapack/lascl.hpp"
#include "lapack/lassq.hpp"

// QR factorization
// ----------------

#include "lapack/geqr2.hpp"
#include "lapack/org2r.hpp"
#include "lapack/orm2r.hpp"
#include "lapack/unmqr.hpp"

// Solution of positive definite systems
// ----------------

#include "lapack/potrf2.hpp"
#include "lapack/potrf.hpp"
#include "lapack/potrs.hpp"

#endif // __TLAPACK_HH__
