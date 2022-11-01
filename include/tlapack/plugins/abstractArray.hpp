/// @file tlapack_abstractArray.hpp
/// @author Weslley S Pereira, University of Colorado Denver, US
///
/// This file contains has two purposes:
///     1. It serves as a template for writing <T>LAPACK abstract arrays.
///     2. It is used by Doxygen to generate the documentation.
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ABSTRACT_ARRAY_HH
#define TLAPACK_ABSTRACT_ARRAY_HH

#include "tlapack/base/arrayTraits.hpp"

namespace tlapack {

    // -------------------------------------------------------------------------
    // Data descriptors for matrices in <T>LAPACK
    
    /**
     * @brief Return the number of rows of a given matrix.
     * 
     * @tparam idx_t    Index type.
     * @tparam matrix_t Matrix type.
     * 
     * @param A Matrix.
     * 
     * @ingroup utils
     */
    template< class idx_t, class matrix_t >
    inline constexpr idx_t
    nrows( const matrix_t& x );

    /**
     * @brief Return the number of columns of a given matrix.
     * 
     * @tparam idx_t    Index type.
     * @tparam matrix_t Matrix type.
     * 
     * @param A Matrix.
     * 
     * @ingroup utils
     */
    template< class idx_t, class matrix_t >
    inline constexpr idx_t
    ncols( const matrix_t& A );

    /**
     * @brief Read policy.
     * 
     * Defines the pairs (i,j) where A(i,j) returns a valid value.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam access_t Access type. One of the following:
     *      MatrixAccessPolicy
     *      dense_t,
     *      upperHessenberg_t,
     *      lowerHessenberg_t,
     *      upperTriangle_t,
     *      lowerTriangle_t,
     *      strictUpper_t,
     *      strictLower_t,
     *      band_t.
     * 
     * @param A Matrix.
     * 
     * @ingroup utils
     */
    template< class access_t, class matrix_t >
    inline constexpr access_t
    read_policy( const matrix_t& A );

    /**
     * @brief Write policy.
     * 
     * Defines the pairs (i,j) where A(i,j) returns a valid reference for
     * reading and writing.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam access_t Access type. One of the following:
     *      MatrixAccessPolicy
     *      dense_t,
     *      upperHessenberg_t,
     *      lowerHessenberg_t,
     *      upperTriangle_t,
     *      lowerTriangle_t,
     *      strictUpper_t,
     *      strictLower_t,
     *      band_t.
     * 
     * @param A Matrix.
     * 
     * @ingroup utils
     */
    template< class access_t, class matrix_t >
    inline constexpr access_t
    write_policy( const matrix_t& A );

    // -------------------------------------------------------------------------
    // Data descriptors for vectors in <T>LAPACK

    /**
     * @brief Return the number of elements of a given vector.
     * 
     * @tparam idx_t    Index type.
     * @tparam vector_t Vector type.
     * 
     * @param v Vector.
     * 
     * @ingroup utils
     */
    template< class idx_t, class vector_t >
    inline constexpr idx_t
    size( const vector_t& v );

    // -------------------------------------------------------------------------
    // Block operations with matrices in <T>LAPACK

    /**
     * @brief Extracts a slice from a given matrix.
     * 
     * @returns a matrix.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam pair_t   Pair of integers.
     *      Stored in pair::first and pair::second.
     * 
     * @param A     Matrix.
     * @param rows  Pair (i,k).
     *      i   is the index of the first row, and
     *      k-1 is the index of the last row.
     * @param cols  Pair (j,l).
     *      j   is the index of the first column, and
     *      l-1 is the index of the last column.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class pair_t >
    inline constexpr auto
    slice( const matrix_t& A, pair_t&& rows, pair_t&& cols );

    /**
     * @brief Extracts a slice from a given matrix.
     * 
     * @returns a vector.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam idx_t    Index type.
     * @tparam pair_t   Pair of integers.
     *      Stored in pair::first and pair::second.
     * 
     * @param A         Matrix.
     * @param rowIdx    Row index.
     * @param cols  Pair (j,l).
     *      j   is the index of the first column, and
     *      l-1 is the index of the last column.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class idx_t, class pair_t >
    inline constexpr auto
    slice( const matrix_t& A, idx_t rowIdx, pair_t&& cols );

    /**
     * @brief Extracts a slice from a given matrix.
     * 
     * @returns a vector.
     * 
     * Default column value is zero. This is useful to obtain vectors from a 
     * given array since `slice( work, {0,n} )` returns a vector of size `n` no
     * matter if `work` is a matrix or a vector.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam pair_t   Pair of integers.
     *      Stored in pair::first and pair::second.
     * @tparam idx_t    Index type.
     * 
     * @param A         Matrix.
     * @param rows      Pair (i,k).
     *      i   is the index of the first row, and
     *      k-1 is the index of the last row.
     * @param colIdx    Column index. Default value is zero.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class pair_t, class idx_t >
    inline constexpr auto
    slice( const matrix_t& A, pair_t&& rows, idx_t colIdx = 0 );
    
    /**
     * @brief Extracts a set of rows from a given matrix.
     * 
     * @returns a matrix.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam pair_t   Pair of integers.
     *      Stored in pair::first and pair::second.
     * 
     * @param A     Matrix.
     * @param rows  Pair (i,k).
     *      i   is the index of the first row, and
     *      k-1 is the index of the last row.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class pair_t >
    inline constexpr auto
    rows( const matrix_t& A, pair_t&& rows );
    
    /**
     * @brief Extracts a set of columns from a given matrix.
     * 
     * @returns a matrix.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam pair_t   Pair of integers.
     *      Stored in pair::first and pair::second.
     * 
     * @param A     Matrix.
     * @param cols  Pair (j,l).
     *      j   is the index of the first column, and
     *      l-1 is the index of the last column.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class pair_t >
    inline constexpr auto
    cols( const matrix_t& A, pair_t&& cols );
    
    /**
     * @brief Extracts a row from a given matrix.
     * 
     * @returns a vector.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam idx_t    Index type.
     * 
     * @param A         Matrix.
     * @param rowIdx    Row index.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class idx_t >
    inline constexpr auto
    row( const matrix_t& A, idx_t rowIdx );
    
    /**
     * @brief Extracts a column from a given matrix.
     * 
     * @returns a vector.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam idx_t    Index type.
     * 
     * @param A         Matrix.
     * @param colIdx    Column index.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class idx_t >
    inline constexpr auto
    col( const matrix_t& A, idx_t colIdx );

    /**
     * @brief Extracts a diagonal from a given matrix.
     * 
     * @returns a vector.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam idx_t    Index type.
     * 
     * @param A         Matrix of size m-by-n. 
     * @param diagIdx   Diagonal index.
     *      - diagIdx = 0: main diagonal.
     *          Vector of size min( m, n ).
     *      - diagIdx < 0: subdiagonal starting on A( -diagIdx, 0 ).
     *          Vector of size min( m, n-diagIdx ) + diagIdx.
     *      - diagIdx > 0: superdiagonal starting on A( 0, diagIdx ).
     *          Vector of size min( m+diagIdx, n ) - diagIdx.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class idx_t >
    inline constexpr auto
    diag( const matrix_t& A, idx_t diagIdx = 0 );

    // -------------------------------------------------------------------------
    // Block operations with vectors in <T>LAPACK

    /**
     * @brief Extracts a slice from a vector.
     * 
     * @returns a vector.
     * 
     * @tparam vector_t Vector type.
     * @tparam pair_t   Pair of integers.
     *      Stored in pair::first and pair::second.
     * 
     * @param v     Vector.
     * @param rows  Pair (i,k).
     *      i   is the index of the first row, and
     *      k-1 is the index of the last row.
     * 
     * @ingroup utils
     */
    template< class vector_t, class pair_t >
    inline constexpr auto
    slice( const vector_t& v, pair_t&& rows );

    // -------------------------------------------------------------------------
    // Create new data based on allocated memory

    /**
     * @see Create class in arrayTraits.hpp
     */
    template< class matrix_t > class Create;

} // namespace tlapack

#endif // TLAPACK_ABSTRACT_ARRAY_HH
