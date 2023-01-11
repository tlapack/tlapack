// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_HH
#define TLAPACK_LEGACY_HH

// =============================================================================
// Template LAPACK

// Auxiliary routines
// ------------------

#include "tlapack/legacy_api/lapack/larf.hpp"
#include "tlapack/legacy_api/lapack/larfg.hpp"
#include "tlapack/legacy_api/lapack/larft.hpp"
#include "tlapack/legacy_api/lapack/larfb.hpp"
#include "tlapack/lapack/lapy2.hpp"
#include "tlapack/lapack/lapy3.hpp"
#include "tlapack/lapack/ladiv.hpp"
#include "tlapack/legacy_api/lapack/laset.hpp"
#include "tlapack/legacy_api/lapack/lacpy.hpp"
#include "tlapack/legacy_api/lapack/lange.hpp"
#include "tlapack/legacy_api/lapack/lanhe.hpp"
#include "tlapack/legacy_api/lapack/lantr.hpp"
#include "tlapack/legacy_api/lapack/lansy.hpp"
#include "tlapack/legacy_api/lapack/larnv.hpp"
#include "tlapack/legacy_api/lapack/lascl.hpp"
#include "tlapack/legacy_api/lapack/lassq.hpp"

// QR factorization
// ----------------

#include "tlapack/legacy_api/lapack/geqr2.hpp"
#include "tlapack/legacy_api/lapack/ung2r.hpp"
#include "tlapack/legacy_api/lapack/unm2r.hpp"
#include "tlapack/legacy_api/lapack/unmqr.hpp"
#include "tlapack/legacy_api/lapack/potrf.hpp"
#include "tlapack/legacy_api/lapack/potrs.hpp"

#endif // TLAPACK_LEGACY_HH
