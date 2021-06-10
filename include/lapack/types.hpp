// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of T-LAPACK.
// T-LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_TYPES_HH__
#define __TLAPACK_TYPES_HH__

#include "blas/types.hpp"

namespace lapack {

// -----------------------------------------------------------------------------
// Use the types from the namespace blas
using blas::size_t;
using blas::int_t;
using blas::Layout;
using blas::Op;
using blas::Uplo;
using blas::Diag;
using blas::Side;
using blas::real_type;
using blas::complex_type;
using blas::scalar_type;

// -----------------------------------------------------------------------------
enum class Sides {
    Left  = 'L',  L = 'L',
    Right = 'R',  R = 'R',
    Both  = 'B',  B = 'B'
};

// -----------------------------------------------------------------------------
enum class Norm {
    One = '1',  // or 'O'
    Two = '2',
    Inf = 'I',
    Fro = 'F',  // or 'E'
    Max = 'M',
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
// larfb
enum class Direction {
    Forward     = 'F',
    Backward    = 'B',
};

// -----------------------------------------------------------------------------
// larfb
enum class StoreV {
    Columnwise  = 'C',
    Rowwise     = 'R',
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

} // namespace lapack

#endif // __TLAPACK_TYPES_HH__