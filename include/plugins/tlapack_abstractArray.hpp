// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_ABSTRACT_ARRAY_HH__
#define __TLAPACK_ABSTRACT_ARRAY_HH__

namespace blas {

    // -----------------------------------------------------------------------------
    // Data traits

    #ifndef TBLAS_ARRAY_TRAITS
        #define TBLAS_ARRAY_TRAITS

        // Data type
        template< class T > struct type_trait {};
        template< class T >
        using type_t = typename type_trait< T >::type;
        
        // Size type
        template< class T > struct sizet_trait {};
        template< class T >
        using size_type = typename sizet_trait< T >::type;

    #endif // TBLAS_ARRAY_TRAITS

    // -----------------------------------------------------------------------------
    // Data description

    // Size
    template< class array_t, class idx_t >
    idx_t size( const array_t& x );

    // Number of rows
    template< class matrix_t, class idx_t >
    idx_t nrows( const matrix_t& x );

    // Number of columns
    template< class matrix_t, class idx_t >
    idx_t ncols( const matrix_t& x );

    // -----------------------------------------------------------------------------
    // Data blocks
    
    // Submatrix
    template< class matrix_t, class SliceSpecRow, class SliceSpecCol >
    matrix_t submatrix( const matrix_t& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept;
    
    // Rows
    template< class matrix_t, class SliceSpec >
    matrix_t rows( const matrix_t& A, SliceSpec&& rows ) noexcept;
    
    // Row
    template< class matrix_t, class vector_t, class idx_t >
    vector_t row( const matrix_t& A, idx_t rowIdx ) noexcept;
    
    // Columns
    template< class matrix_t, class SliceSpec >
    matrix_t cols( const matrix_t& A, SliceSpec&& rows ) noexcept;
    
    // Column
    template< class matrix_t, class vector_t, class idx_t >
    vector_t col( const matrix_t& A, idx_t colIdx ) noexcept;

    // Subvector
    template< class vector_t, class SliceSpec >
    vector_t subvector( const vector_t& v, SliceSpec&& rows ) noexcept;

    // Diagonal
    template< class matrix_t, class vector_t, class int_t >
    vector_t diag( const matrix_t& A, int_t diagIdx = 0 ) noexcept;

} // namespace blas

namespace lapack {
    
    using blas::type_t;
    using blas::size_type;

    using blas::size;
    using blas::nrows;
    using blas::ncols;

    using blas::submatrix;
    using blas::rows;
    using blas::row;
    using blas::cols;
    using blas::col;
    using blas::subvector;
    using blas::diag;

} // namespace lapack

#endif // __TLAPACK_ABSTRACT_ARRAY_HH__