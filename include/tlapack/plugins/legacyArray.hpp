// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACYARRAY_HH
#define TLAPACK_LEGACYARRAY_HH

#include <cassert>

#include "tlapack/base/legacyArray.hpp"
#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/workspace.hpp"

namespace tlapack {

    // -----------------------------------------------------------------------------
    // Data traits

    // Layout
    template< typename T, class idx_t, Layout L >
    constexpr Layout layout< legacyMatrix<T,idx_t,L> > = L;

    // Layout
    template< typename T, class idx_t >
    constexpr Layout layout< legacyBandedMatrix<T,idx_t> > = Layout::BandStorage;

    // Transpose type
    template< class T, class idx_t >
    struct transpose_type_trait< legacyMatrix<T,idx_t,Layout::ColMajor> > {
        using type = legacyMatrix<T,idx_t,Layout::RowMajor>;
    };
    template< class T, class idx_t >
    struct transpose_type_trait< legacyMatrix<T,idx_t,Layout::RowMajor> > {
        using type = legacyMatrix<T,idx_t,Layout::ColMajor>;
    };

    // -----------------------------------------------------------------------------
    // Data description

    // Number of rows
    template< typename T, class idx_t, Layout layout >
    inline constexpr auto
    nrows( const legacyMatrix<T,idx_t,layout>& A ){ return A.m; }

    // Number of columns
    template< typename T, class idx_t, Layout layout >
    inline constexpr auto
    ncols( const legacyMatrix<T,idx_t,layout>& A ){ return A.n; }

    // Read policy
    template< typename T, class idx_t, Layout layout >
    inline constexpr auto
    read_policy( const legacyMatrix<T,idx_t,layout>& A ) {
        return dense;
    }

    // Write policy
    template< typename T, class idx_t, Layout layout >
    inline constexpr auto
    write_policy( const legacyMatrix<T,idx_t,layout>& A ) {
        return dense;
    }

    // Size
    template< typename T, class idx_t, typename int_t, Direction direction >
    inline constexpr auto
    size( const legacyVector<T,idx_t,int_t,direction>& x ){ return x.n; }

    // Number of rows
    template< typename T, class idx_t >
    inline constexpr auto
    nrows( const legacyBandedMatrix<T,idx_t>& A ){ return A.m; }

    // Number of columns
    template< typename T, class idx_t >
    inline constexpr auto
    ncols( const legacyBandedMatrix<T,idx_t>& A ){ return A.n; }

    // Lowerband
    template< typename T, class idx_t >
    inline constexpr auto
    lowerband( const legacyBandedMatrix<T,idx_t>& A ){ return A.kl; }

    // Upperband
    template< typename T, class idx_t >
    inline constexpr auto
    upperband( const legacyBandedMatrix<T,idx_t>& A ){ return A.ku; }

    // Read policy
    template< typename T, class idx_t >
    inline constexpr auto
    read_policy( const legacyBandedMatrix<T,idx_t>& A ) {
        return band_t {
            (std::size_t) A.kl, (std::size_t) A.ku
        };
    }

    // Access policy
    template< typename T, class idx_t >
    inline constexpr auto
    write_policy( const legacyBandedMatrix<T,idx_t>& A ) {
        return band_t {
            (std::size_t) A.kl, (std::size_t) A.ku
        };
    }

    // -----------------------------------------------------------------------------
    // Data blocks

    #define isSlice(SliceSpec) !std::is_convertible< SliceSpec, idx_t >::value
    
    // Slice
    template< typename T, class idx_t, Layout layout, class SliceSpecRow, class SliceSpecCol,
        typename std::enable_if< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int >::type = 0
    >
    inline constexpr auto
    slice( const legacyMatrix<T,idx_t,layout>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept {
        assert( rows.first >= 0 and rows.first < nrows(A));
        assert( rows.second >= 0 and rows.second <= nrows(A));
        assert( rows.first <= rows.second );
        assert( cols.first >= 0 and cols.first < ncols(A));
        assert( cols.second >= 0 and cols.second <= ncols(A));
        assert( cols.first <= cols.second );
        return legacyMatrix<T,idx_t,layout>(
            rows.second-rows.first, cols.second-cols.first,
            &A(rows.first,cols.first), A.ldim
        );
    }

    #undef isSlice
    
    // Slice
    template< typename T, class idx_t, Layout layout, class SliceSpecCol >
    inline constexpr auto
    slice( const legacyMatrix<T,idx_t,layout>& A, size_type<legacyMatrix<T,idx_t,layout>> rowIdx, SliceSpecCol&& cols ) noexcept {
        assert( cols.first >= 0 and cols.first < ncols(A));
        assert( cols.second >= 0 and cols.second <= ncols(A));
        assert( cols.first <= cols.second );
        assert( rowIdx >= 0 and rowIdx < nrows(A));
        return legacyVector<T,idx_t,idx_t>( cols.second-cols.first, &A(rowIdx,cols.first), layout == Layout::ColMajor ? A.ldim : 1 );
    }
    
    // Slice
    template< typename T, class idx_t, Layout layout, class SliceSpecRow >
    inline constexpr auto
    slice( const legacyMatrix<T,idx_t,layout>& A, SliceSpecRow&& rows, size_type<legacyMatrix<T,idx_t,layout>> colIdx = 0 ) noexcept {
        assert( rows.first >= 0 and rows.first < nrows(A));
        assert( rows.second >= 0 and rows.second <= nrows(A));
        assert( rows.first <= rows.second );
        assert( colIdx >= 0 and colIdx < ncols(A));
        return legacyVector<T,idx_t,idx_t>( rows.second-rows.first, &A(rows.first,colIdx), layout == Layout::RowMajor ? A.ldim : 1 );
    }
    
    // Rows
    template< typename T, class idx_t, Layout layout, class SliceSpec >
    inline constexpr auto
    rows( const legacyMatrix<T,idx_t,layout>& A, SliceSpec&& rows ) noexcept {
        assert( rows.first >= 0 and rows.first < nrows(A));
        assert( rows.second >= 0 and rows.second <= nrows(A));
        assert( rows.first <= rows.second );
        return legacyMatrix<T,idx_t,layout>(
            rows.second-rows.first, A.n,
            &A(rows.first,0), A.ldim
        );
    }
    
    // Row
    template< typename T, class idx_t >
    inline constexpr auto
    row( const legacyMatrix<T,idx_t>& A, size_type<legacyMatrix<T,idx_t>> rowIdx ) noexcept {
        assert( rowIdx >= 0 and rowIdx < nrows(A));
        return legacyVector<T,idx_t,idx_t>( A.n, &A(rowIdx,0), A.ldim );
    }
    
    // Row
    template< typename T, class idx_t >
    inline constexpr auto
    row( const legacyMatrix<T,idx_t,Layout::RowMajor>& A, size_type<legacyMatrix<T,idx_t,Layout::RowMajor>> rowIdx ) noexcept {
        assert( rowIdx >= 0 and rowIdx < nrows(A));
        return legacyVector<T,idx_t>( A.n, &A(rowIdx,0) );
    }
    
    // Columns
    template< typename T, class idx_t, Layout layout, class SliceSpec >
    inline constexpr auto
    cols( const legacyMatrix<T,idx_t,layout>& A, SliceSpec&& cols ) noexcept {
        assert( cols.first >= 0 and cols.first < ncols(A));
        assert( cols.second >= 0 and cols.second <= ncols(A));
        return legacyMatrix<T,idx_t,layout>(
            A.m, cols.second-cols.first,
            &A(0,cols.first), A.ldim
        );
    }
    
    // Column
    template< typename T, class idx_t >
    inline constexpr auto
    col( const legacyMatrix<T,idx_t>& A, size_type<legacyMatrix<T,idx_t>> colIdx ) noexcept {
        assert( colIdx >= 0 and colIdx < ncols(A));
        return legacyVector<T,idx_t>( A.m, &A(0,colIdx) );
    }
    
    // Column
    template< typename T, class idx_t >
    inline constexpr auto
    col( const legacyMatrix<T,idx_t,Layout::RowMajor>& A, size_type<legacyMatrix<T,idx_t,Layout::RowMajor>> colIdx ) noexcept {
        assert( colIdx >= 0 and colIdx < ncols(A));
        return legacyVector<T,idx_t,idx_t>( A.m, &A(0,colIdx), A.ldim );
    }

    // Diagonal
    template< typename T, class idx_t, Layout layout >
    inline constexpr auto
    diag( const legacyMatrix<T,idx_t,layout>& A, int diagIdx = 0 ) noexcept {
        
        T* ptr  = (diagIdx >= 0) ? &A(0,diagIdx) : &A(-diagIdx,0);
        idx_t n = (diagIdx >= 0)
                    ? std::min( A.m+diagIdx, A.n ) - (idx_t) diagIdx
                    : std::min( A.m, A.n-diagIdx ) + (idx_t) diagIdx;
        
        return legacyVector<T,idx_t,idx_t>( n, ptr, A.ldim + 1 );
    }

    // slice
    template< typename T, class idx_t, typename int_t, Direction direction, class SliceSpec >
    inline constexpr auto
    slice( const legacyVector<T,idx_t,int_t,direction>& v, SliceSpec&& rows ) noexcept {
        assert( rows.first >= 0 and rows.first < size(v));
        assert( rows.second >= 0 and rows.second <= size(v));
        return legacyVector<T,idx_t,int_t,direction>( rows.second-rows.first, &v[rows.first], v.inc );
    }

    // -----------------------------------------------------------------------------
    // Create objects

    template< class T, class idx_t, Layout layout >
    struct CreateImpl< legacyMatrix<T,idx_t,layout>, int > {

        using matrix_t = legacyMatrix<T,idx_t,layout>;

        inline constexpr auto
        operator()( std::vector<T>& v, idx_t m, idx_t n ) const {
            assert( m >= 0 && n >= 0 );
            v.resize( m*n ); // Allocates space in memory
            return matrix_t( m, n, v.data() );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n, Workspace& rW ) const
        {
            assert( m >= 0 && n >= 0 );
            rW = ( layout == Layout::ColMajor ) ? W.extract( m*sizeof(T), n )
                                                : W.extract( n*sizeof(T), m );
            return ( W.isContiguous() )
                ? matrix_t( m, n, (T*) W.data() ) // contiguous space in memory
                : matrix_t( m, n, (T*) W.data(), W.getLdim()/sizeof(T) );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n ) const
        {
            assert( m >= 0 && n >= 0 );
            tlapack_check( ( layout == Layout::ColMajor ) ? W.contains( m*sizeof(T), n )
                                                          : W.contains( n*sizeof(T), m ) );
            return ( W.isContiguous() )
                ? matrix_t( m, n, (T*) W.data() ) // contiguous space in memory
                : matrix_t( m, n, (T*) W.data(), W.getLdim()/sizeof(T) );
        }
    };

    template< class T, class idx_t, typename int_t, Direction D >
    struct CreateImpl< legacyVector<T,idx_t,int_t,D>, int > {

        using vector_t = legacyVector<T,idx_t,int_t,D>;

        inline constexpr auto
        operator()( std::vector<T>& v, idx_t m ) const {
            assert( m >= 0 );
            v.resize( m ); // Allocates space in memory
            return vector_t( m, v.data() );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, Workspace& rW ) const
        {
            assert( m >= 0 );
            rW = W.extract( sizeof(T), m );
            return ( W.isContiguous() )
                ? vector_t( m, (T*) W.data() ) // contiguous space in memory
                : vector_t( m, (T*) W.data(), W.getLdim()/sizeof(T) );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m ) const
        {
            assert( m >= 0 );
            tlapack_check( W.contains( sizeof(T), m ) );
            return ( W.isContiguous() )
                ? vector_t( m, (T*) W.data() ) // contiguous space in memory
                : vector_t( m, (T*) W.data(), W.getLdim()/sizeof(T) );
        }
    };

    // -----------------------------------------------------------------------------
    // Cast to Legacy arrays

    template< typename T, class idx_t, Layout layout >
    inline constexpr auto
    legacy_matrix( const legacyMatrix<T,idx_t,layout>& A ) noexcept { return A; }

    template< class T, class idx_t, class int_t, Direction direction >
    inline constexpr auto
    legacy_matrix( const legacyVector<T,idx_t,int_t,direction>& v ) noexcept {
        return legacyMatrix<T,idx_t,Layout::ColMajor>( 1, v.n, v.ptr, v.inc );
    }

    template< class T, class idx_t, Direction direction >
    inline constexpr auto
    legacy_matrix( const legacyVector<T,idx_t,internal::StrongOne,direction>& v ) noexcept {
        return legacyMatrix<T,idx_t,Layout::ColMajor>( v.n, 1, v.ptr );
    }

    template< typename T, class idx_t, typename int_t, Direction direction >
    inline constexpr auto
    legacy_vector( const legacyVector<T,idx_t,int_t,direction>& v ) noexcept { return v; }

    // Matrix and vector type specialization:

    #ifndef TLAPACK_PREFERRED_MATRIX

        // for two types
        // should be especialized for every new matrix class
        template< typename matrixA_t, typename matrixB_t >
        struct matrix_type_traits< matrixA_t, matrixB_t >
        {
            using T = scalar_type< type_t<matrixA_t>, type_t<matrixB_t> >;
            using idx_t = size_type<matrixA_t>;

            static constexpr Layout LA = layout<matrixA_t>;
            static constexpr Layout LB = layout<matrixB_t>;
            static constexpr Layout L  =
                ((LA == Layout::RowMajor) && (LB == Layout::RowMajor))
                    ? Layout::RowMajor
                    : Layout::ColMajor;

            using type = legacyMatrix<T,idx_t,L>;
        };

        // for two types
        // should be especialized for every new vector class
        template< typename vecA_t, typename vecB_t >
        struct vector_type_traits< vecA_t, vecB_t >
        {
            using T = scalar_type< type_t<vecA_t>, type_t<vecB_t> >;
            using idx_t = size_type<vecA_t>;

            using type = legacyVector<T,idx_t,idx_t>;
        };

    #endif // TLAPACK_PREFERRED_MATRIX

    /// TODO: Complete the implementation of vector_type_traits and matrix_type_traits

} // namespace tlapack

#endif // TLAPACK_LEGACYARRAY_HH
