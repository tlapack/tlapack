// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
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

#include "legacy_api/lapack/larf.hpp"
#include "legacy_api/lapack/larfg.hpp"
#include "legacy_api/lapack/larft.hpp"
#include "legacy_api/lapack/larfb.hpp"
#include "lapack/lapy2.hpp"
#include "lapack/lapy3.hpp"
#include "lapack/ladiv.hpp"
#include "legacy_api/lapack/laset.hpp"
#include "legacy_api/lapack/lacpy.hpp"
#include "legacy_api/lapack/lange.hpp"
#include "legacy_api/lapack/lanhe.hpp"
#include "legacy_api/lapack/lantr.hpp"
#include "legacy_api/lapack/lansy.hpp"
#include "legacy_api/lapack/larnv.hpp"
#include "legacy_api/lapack/lascl.hpp"
#include "legacy_api/lapack/lassq.hpp"

// QR factorization
// ----------------

#include "legacy_api/lapack/geqr2.hpp"
#include "legacy_api/lapack/ung2r.hpp"
#include "legacy_api/lapack/unm2r.hpp"
#include "legacy_api/lapack/unmqr.hpp"
#include "legacy_api/lapack/potrf.hpp"
#include "legacy_api/lapack/potrs.hpp"

#endif // TLAPACK_LEGACY_HH
