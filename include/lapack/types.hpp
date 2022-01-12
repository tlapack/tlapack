// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_TYPES_HH__
#define __TLAPACK_TYPES_HH__

#include "blas/types.hpp"

namespace lapack {

// -----------------------------------------------------------------------------
// Use the types from the namespace blas

using blas::idx_t;
using blas::int_t;
using blas::Layout;
using blas::Op;
using blas::Uplo;
using blas::Diag;
using blas::Side;
using blas::real_type;
using blas::complex_type;
using blas::scalar_type;
using blas::is_complex;
using blas::type_t;
using blas::size_type;
using blas::enable_if_t;
using blas::is_convertible_v;
using blas::is_same_v;
using blas::zero_t;

// -----------------------------------------------------------------------------
// Diagonal matrices

struct nonUnit_diagonal_t {
    constexpr operator blas::Diag() const { return blas::Diag::NonUnit; }
};
struct unit_diagonal_t {
    constexpr operator blas::Diag() const { return blas::Diag::Unit; }
};

// constants
constexpr nonUnit_diagonal_t nonUnit_diagonal = { };
constexpr unit_diagonal_t unit_diagonal = { };

// -----------------------------------------------------------------------------
// Operations over data

struct noTranspose_t {
    constexpr operator blas::Op() const { return blas::Op::NoTrans; }
};
struct transpose_t {
    constexpr operator blas::Op() const { return blas::Op::Trans; }
};
struct conjTranspose_t {
    constexpr operator blas::Op() const { return blas::Op::ConjTrans; }
};

// Constants
constexpr noTranspose_t noTranspose = { };
constexpr transpose_t transpose = { };
constexpr conjTranspose_t conjTranspose = { };

// -----------------------------------------------------------------------------
// Matrix structure types

// Full matrix type
struct general_matrix_t {
    constexpr operator blas::Uplo() const { return blas::Uplo::General; }
};

// Upper triangle type
struct upper_triangle_t {
    constexpr operator blas::Uplo() const { return blas::Uplo::Upper; }
};

// Lower triangle type
struct lower_triangle_t {
    constexpr operator blas::Uplo() const { return blas::Uplo::Lower; }
};

// Hessenberg matrix type
struct hessenberg_matrix_t { };

// Band matrix type
template< std::size_t kl, std::size_t ku >
struct band_matrix_t {
    static constexpr std::size_t lower_bandwidth = kl;
    static constexpr std::size_t upper_bandwidth = ku;
};

// Symmetric lower band matrix type
template< std::size_t k >
struct symmetric_lowerband_t {
    static constexpr std::size_t bandwidth = k;
};

// Symmetric upper band matrix type
template< std::size_t k >
struct symmetric_upperband_t {
    static constexpr std::size_t bandwidth = k;
};

// Constants
constexpr general_matrix_t general_matrix = { };
constexpr upper_triangle_t upper_triangle = { };
constexpr lower_triangle_t lower_triangle = { };
constexpr hessenberg_matrix_t hessenberg_matrix = { };

// -----------------------------------------------------------------------------
// Norm types

struct max_norm_t { };
struct one_norm_t { };
struct inf_norm_t { };
struct frob_norm_t { };

// Constants
constexpr max_norm_t max_norm = { };
constexpr one_norm_t one_norm = { };
constexpr inf_norm_t inf_norm = { };
constexpr frob_norm_t frob_norm = { };

// -----------------------------------------------------------------------------
// Directions

struct forward_t { };
struct backward_t { };

// Constants
constexpr forward_t forward { };
constexpr backward_t backward { };

// -----------------------------------------------------------------------------
// Storage types

struct columnwise_storage_t { };
struct rowwise_storage_t { };

// Constants
constexpr columnwise_storage_t columnwise_storage { };
constexpr rowwise_storage_t rowwise_storage { };

// -----------------------------------------------------------------------------
// Sides

struct left_side_t {
    constexpr operator blas::Side() const { return blas::Side::Left; }
};
struct right_side_t {
    constexpr operator blas::Side() const { return blas::Side::Right; }
};

// Constants
constexpr left_side_t left_side { };
constexpr right_side_t right_side { };

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