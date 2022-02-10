// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_ARRAY_HH__
#define __TLAPACK_LEGACY_ARRAY_HH__

// -----------------------------------------------------------------------------
// Integer types BLAS_SIZE_T and BLAS_INT_T

#include <cstdint> // Defines std::int64_t
#include <cstddef> // Defines std::size_t

#ifndef BLAS_SIZE_T
    #define BLAS_SIZE_T std::size_t
#endif

#ifndef BLAS_INT_T
    #define BLAS_INT_T std::int64_t
#endif
// -----------------------------------------------------------------------------

namespace blas {

    /// legacyMatrix
    template< typename T, Layout layout = Layout::ColMajor >
    struct legacyMatrix {
        using idx_t = BLAS_SIZE_T;
        idx_t m, n, ldim;
        T* ptr;
        
        template< enable_if< layout == Layout::ColMajor ,int>=0 >
        inline constexpr T&
        operator()( idx_t i, idx_t j ) const noexcept { return ptr[ i + j*ldim ]; }
        
        template< enable_if< layout == Layout::RowMajor ,int>=0 >
        inline constexpr T&
        operator()( idx_t i, idx_t j ) const noexcept { return ptr[ i*ldim + j ]; }
        
        inline constexpr legacyMatrix( idx_t m, idx_t n, T* ptr, idx_t ldim )
        : m(m), n(n), ptr(ptr), ldim(ldim)
        {
            blas_error_if( m < 0 );
            blas_error_if( n < 0 );
            blas_error_if( ldim < ((layout == Layout::ColMajor) ? m : n) );
        }
    };

    // template< typename T >
    // struct legacyMatrix {
    //     idx_t m, n, ldim;
    //     T* ptr;
        
    //     inline constexpr T&
    //     operator()( idx_t i, idx_t j ) const noexcept { return ptr[ i + j*ldim ]; }
        
    //     inline constexpr legacyMatrix( idx_t m, idx_t n, T* ptr, idx_t ldim )
    //     : m(m), n(n), ptr(ptr), ldim(ldim)
    //     {
    //         blas_error_if( m < 0 );
    //         blas_error_if( n < 0 );
    //         blas_error_if( ldim < m );
    //     }
    // };

    // /// legacyRowMajorMatrix
    // template< typename T >
    // struct legacyRowMajorMatrix {
    //     idx_t m, n, ldim;
    //     T* ptr;
        
    //     inline constexpr T&
    //     operator()( idx_t i, idx_t j ) const noexcept { return ptr[ i*ldim + j ]; }
        
    //     inline constexpr legacyRowMajorMatrix( idx_t m, idx_t n, T* ptr, idx_t ldim )
    //     : m(m), n(n), ptr(ptr), ldim(ldim)
    //     {
    //         blas_error_if( m < 0 );
    //         blas_error_if( n < 0 );
    //         blas_error_if( ldim < n );
    //     }
    // };

    /// legacyVector: assumes increment is always positive
    template< typename T, Direction direction = Direction::Forward >
    struct legacyVector {
        using idx_t = BLAS_SIZE_T;
        idx_t n, inc;
        T* ptr;
        
        template< enable_if< direction == Direction::Forward ,int>=0 >
        inline constexpr T&
        operator[]( idx_t i ) const noexcept { return ptr[ i*inc ]; }
        
        template< enable_if< direction == Direction::Backward ,int>=0 >
        inline constexpr T&
        operator[]( idx_t i ) const noexcept { return ptr[ ((n-1)-i)*inc ]; }
        
        inline constexpr legacyVector( idx_t n, T* ptr, idx_t inc )
        : n(n), ptr(ptr), inc(inc)
        {
            blas_error_if( n < 0 );
            blas_error_if( inc <= 0 );
        }
    };

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

    // Data type
    template< typename T, Layout layout >
    struct type_trait< legacyMatrix<T,layout> > { using type = T; };
    // Size type
    template< typename T, Layout layout >
    struct sizet_trait< legacyMatrix<T,layout> > { using type = typename legacyMatrix<T,layout>::idx_t; };

    // Data type
    template< typename T >
    struct type_trait< legacyVector<T> > { using type = T; };
    // Size type
    template< typename T >
    struct sizet_trait< legacyVector<T> > { using type = typename legacyVector<T>::idx_t; };

    // -----------------------------------------------------------------------------
    // Data description

    // Size
    template< typename T, Layout layout >
    inline constexpr auto
    size( const legacyMatrix<T,layout>& A ){ return A.m*A.n; }

    // Number of rows
    template< typename T, Layout layout >
    inline constexpr auto
    nrows( const legacyMatrix<T,layout>& A ){ return A.m }

    // Number of columns
    template< typename T, Layout layout >
    inline constexpr auto
    ncols( const legacyMatrix<T,layout>& A ){ return A.n; }

    // Size
    template< typename T >
    inline constexpr auto
    size( const legacyVector<T>& x ){ return x.n; }

    // -----------------------------------------------------------------------------
    // Data blocks
    
    // Submatrix
    template< typename T, Layout layout, class SliceSpecRow, class SliceSpecCol >
    inline constexpr auto
    submatrix( const legacyMatrix<T,layout>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept {
        return legacyMatrix(
            rows.second-rows.first, cols.second-cols.first,
            &A(rows.first,cols.first), A.ldim
        );
    }
    
    // Rows
    template< typename T, Layout layout, class SliceSpec >
    inline constexpr auto
    rows( const legacyMatrix<T,layout>& A, SliceSpec&& rows ) noexcept {
        return legacyMatrix( rows.second-rows.first, A.n, &A(rows.first,0), A.ldim );
    }
    
    // Row
    template< typename T, Layout layout >
    inline constexpr auto
    row( const legacyMatrix<T,layout>& A, typename legacyMatrix<T,layout>::idx_t rowIdx ) noexcept {
        return legacyVector( A.n, &A(rowIdx,0), (layout == Layout::ColMajor) ? (int_t)A.ldim : 1 );
    }
    
    // Columns
    template< typename T, Layout layout, class SliceSpec >
    inline constexpr auto
    cols( const legacyMatrix<T,layout>& A, SliceSpec&& rows ) noexcept {
        return legacyMatrix( A.m, cols.second-cols.first, &A(0,cols.first), A.ldim );
    }
    
    // Column
    template< typename T, Layout layout >
    inline constexpr auto
    col( const legacyMatrix<T,layout>& A, typename legacyMatrix<T,layout>::idx_t colIdx ) noexcept {
        return legacyVector( A.m, &A(0,colIdx), (layout == Layout::ColMajor) ? 1 : (int_t)A.ldim );
    }

    // Diagonal
    template< typename T, Layout layout >
    inline constexpr auto
    diag( const legacyMatrix<T,layout>& A, int diagIdx = 0 ) noexcept {
        using idx_t = typename legacyMatrix<T,layout>::idx_t;
        using int_t = typename legacyVector<T>::int_t;
        T* ptr  = (diagIdx >= 0) ? &A(0,diagIdx) : &A(-diagIdx,0);
        idx_t n = std::min(A.m,A.n) - (idx_t)( (diagIdx >= 0) ? diagIdx : -diagIdx );
        return legacyVector( std::move(n), std::move(ptr), (int_t)A.ldim + 1 );
    }

    // Subvector
    template< typename T, class SliceSpec >
    inline constexpr auto
    subvector( const legacyVector<T>& v, SliceSpec&& rows ) noexcept {
        return legacyVector( rows.second-rows.first, &v[rows.first], v.inc );
    }
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
    using blas::diag;

    using blas::subvector;

} // namespace lapack

#endif // __TLAPACK_LEGACY_ARRAY_HH__