/// @file types.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_TYPES_HH
#define TLAPACK_LEGACY_TYPES_HH

#include "tlapack/base/types.hpp"

// -----------------------------------------------------------------------------
// Integer types TLAPACK_SIZE_T and TLAPACK_INT_T

#include <cstdint> // Defines std::int64_t
#include <cstddef> // Defines std::size_t

#ifdef USE_LAPACKPP_WRAPPERS
    #ifndef TLAPACK_SIZE_T
        #define TLAPACK_SIZE_T std::int64_t
    #endif
#else
    #ifndef TLAPACK_SIZE_T
        #define TLAPACK_SIZE_T std::size_t
    #endif
#endif

#ifndef TLAPACK_INT_T
    #define TLAPACK_INT_T std::int64_t
#endif
// -----------------------------------------------------------------------------

namespace tlapack {
    
using idx_t = TLAPACK_SIZE_T;
using int_t = TLAPACK_INT_T;

// -----------------------------------------------------------------------------
enum class Sides {
    Left  = 'L',  L = 'L',
    Right = 'R',  R = 'R',
    Both  = 'B',  B = 'B'
};

// -----------------------------------------------------------------------------
// Job for computing eigenvectors and singular vectors
// # needs custom map
enum class Job {
    NoVec        = 'N',
    Vec          = 'V',  // geev, syev, ...
    UpdateVec    = 'U',  // gghrd#, hbtrd, hgeqz#, hseqr#, ... (many compq or compz)

    AllVec       = 'A',  // gesvd, gesdd, gejsv#
    SomeVec      = 'S',  // gesvd, gesdd, gejsv#, gesvj#
    OverwriteVec = 'O',  // gesvd, gesdd

    CompactVec   = 'P',  // bdsdc
    SomeVecTol   = 'C',  // gesvj
    VecJacobi    = 'J',  // gejsv
    Workspace    = 'W',  // gejsv
};

// -----------------------------------------------------------------------------
// hseqr
enum class JobSchur {
    Eigenvalues  = 'E',
    Schur        = 'S',
};

// -----------------------------------------------------------------------------
// gees
// todo: generic yes/no
enum class Sort {
    NotSorted   = 'N',
    Sorted      = 'S',
};

// -----------------------------------------------------------------------------
// syevx
enum class Range {
    All         = 'A',
    Value       = 'V',
    Index       = 'I',
};

// -----------------------------------------------------------------------------
enum class Vect {
    Q           = 'Q',  // orgbr, ormbr
    P           = 'P',  // orgbr, ormbr
    None        = 'N',  // orgbr, ormbr, gbbrd
    Both        = 'B',  // orgbr, ormbr, gbbrd
};

// -----------------------------------------------------------------------------
// lascl
enum class MatrixType {
    General     = 'G',
    Lower       = 'L',
    Upper       = 'U',
    Hessenberg  = 'H',
    LowerBand   = 'B',
    UpperBand   = 'Q',
    Band        = 'Z',
};

// -----------------------------------------------------------------------------
// trevc
enum class HowMany {
    All           = 'A',
    Backtransform = 'B',
    Select        = 'S',
};

// -----------------------------------------------------------------------------
// *svx, *rfsx
enum class Equed {
    None        = 'N',
    Row         = 'R',
    Col         = 'C',
    Both        = 'B',
    Yes         = 'Y',  // porfsx
};

// -----------------------------------------------------------------------------
// *svx
// todo: what's good name for this?
enum class Factored {
    Factored    = 'F',
    NotFactored = 'N',
    Equilibrate = 'E',
};

// -----------------------------------------------------------------------------
// geesx, trsen
enum class Sense {
    None        = 'N',
    Eigenvalues = 'E',
    Subspace    = 'V',
    Both        = 'B',
};

// -----------------------------------------------------------------------------
// disna
enum class JobCond {
    EigenVec         = 'E',
    LeftSingularVec  = 'L',
    RightSingularVec = 'R',
};

// -----------------------------------------------------------------------------
// {ge,gg}{bak,bal}
enum class Balance {
    None        = 'N',
    Permute     = 'P',
    Scale       = 'S',
    Both        = 'B',
};

// -----------------------------------------------------------------------------
// stebz, larrd, stein docs
enum class Order {
    Block       = 'B',
    Entire      = 'E',
};

// -----------------------------------------------------------------------------
// check_ortho (LAPACK testing zunt01)
enum class RowCol {
    Col = 'C',
    Row = 'R',
};

} // namespace tlapack

#endif // TLAPACK_LEGACY_TYPES_HH
