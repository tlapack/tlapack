// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACYARRAY_HH__
#define __TLAPACK_LEGACYARRAY_HH__

#include <utility>
#include <type_traits>
#include <cassert>

#include "legacy_api/legacyArray.hpp"
#include "base/arrayTraits.hpp"

namespace tlapack {

    // -----------------------------------------------------------------------------
    // Data traits

    // Layout
    template< typename T, Layout L >
    constexpr Layout layout< legacyMatrix<T,L> > = L;

    // Layout
    template< typename T >
    constexpr Layout layout< legacyBandedMatrix<T> > = Layout::BandStorage;

    // -----------------------------------------------------------------------------
    // Data description

    // Number of rows
    template< typename T, Layout layout >
    inline constexpr auto
    nrows( const legacyMatrix<T,layout>& A ){ return A.m; }

    // Number of columns
    template< typename T, Layout layout >
    inline constexpr auto
    ncols( const legacyMatrix<T,layout>& A ){ return A.n; }

    // Read policy
    template< typename T, Layout layout >
    inline constexpr auto
    read_policy( const legacyMatrix<T,layout>& A ) {
        return dense;
    }

    // Write policy
    template< typename T, Layout layout >
    inline constexpr auto
    write_policy( const legacyMatrix<T,layout>& A ) {
        return dense;
    }

    // Size
    template< typename T, typename int_t, Direction direction >
    inline constexpr auto
    size( const legacyVector<T,int_t,direction>& x ){ return x.n; }

    // Number of rows
    template< typename T >
    inline constexpr auto
    nrows( const legacyBandedMatrix<T>& A ){ return A.m; }

    // Number of columns
    template< typename T >
    inline constexpr auto
    ncols( const legacyBandedMatrix<T>& A ){ return A.n; }

    // Lowerband
    template< typename T >
    inline constexpr auto
    lowerband( const legacyBandedMatrix<T>& A ){ return A.kl; }

    // Upperband
    template< typename T >
    inline constexpr auto
    upperband( const legacyBandedMatrix<T>& A ){ return A.ku; }

    // Read policy
    template< typename T >
    inline constexpr auto
    read_policy( const legacyBandedMatrix<T>& A ) {
        return band_t {
            (std::size_t) A.kl, (std::size_t) A.ku
        };
    }

    // Access policy
    template< typename T >
    inline constexpr auto
    write_policy( const legacyBandedMatrix<T>& A ) {
        return band_t {
            (std::size_t) A.kl, (std::size_t) A.ku
        };
    }

    // -----------------------------------------------------------------------------
    // Data blocks

    #define isSlice(SliceSpec) !std::is_convertible< SliceSpec, typename legacyMatrix<T>::idx_t >::value
    
    // Slice
    template< typename T, Layout layout, class SliceSpecRow, class SliceSpecCol,
        typename std::enable_if< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int >::type = 0
    >
    inline constexpr auto
    slice( const legacyMatrix<T,layout>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept {
        assert( rows.first >= 0 and rows.first < nrows(A));
        assert( rows.second >= 0 and rows.second <= nrows(A));
        assert( rows.first <= rows.second );
        assert( cols.first >= 0 and cols.first < ncols(A));
        assert( cols.second >= 0 and cols.second <= ncols(A));
        assert( cols.first <= cols.second );
        return legacyMatrix<T,layout>(
            rows.second-rows.first, cols.second-cols.first,
            &A(rows.first,cols.first), A.ldim
        );
    }

    #undef isSlice
    
    // Slice
    template< typename T, Layout layout, class SliceSpecCol >
    inline constexpr auto
    slice( const legacyMatrix<T,layout>& A, typename legacyMatrix<T>::idx_t rowIdx, SliceSpecCol&& cols ) noexcept {
        assert( cols.first >= 0 and cols.first < ncols(A));
        assert( cols.second >= 0 and cols.second <= ncols(A));
        assert( cols.first <= cols.second );
        assert( rowIdx >= 0 and rowIdx < nrows(A));
        using idx_t = typename legacyMatrix<T>::idx_t;
        return legacyVector<T,idx_t>( cols.second-cols.first, &A(rowIdx,cols.first), layout == Layout::ColMajor ? A.ldim : 1 );
    }
    
    // Slice
    template< typename T, Layout layout, class SliceSpecRow >
    inline constexpr auto
    slice( const legacyMatrix<T,layout>& A, SliceSpecRow&& rows, typename legacyMatrix<T>::idx_t colIdx = 0 ) noexcept {
        assert( rows.first >= 0 and rows.first < nrows(A));
        assert( rows.second >= 0 and rows.second <= nrows(A));
        assert( rows.first <= rows.second );
        assert( colIdx >= 0 and colIdx < ncols(A));
        using idx_t = typename legacyMatrix<T>::idx_t;
        return legacyVector<T,idx_t>( rows.second-rows.first, &A(rows.first,colIdx), layout == Layout::RowMajor ? A.ldim : 1 );
    }
    
    // Rows
    template< typename T, Layout layout, class SliceSpec >
    inline constexpr auto
    rows( const legacyMatrix<T,layout>& A, SliceSpec&& rows ) noexcept {
        assert( rows.first >= 0 and rows.first < nrows(A));
        assert( rows.second >= 0 and rows.second <= nrows(A));
        assert( rows.first <= rows.second );
        return legacyMatrix<T,layout>(
            rows.second-rows.first, A.n,
            &A(rows.first,0), A.ldim
        );
    }
    
    // Row
    template< typename T >
    inline constexpr auto
    row( const legacyMatrix<T>& A, typename legacyMatrix<T>::idx_t rowIdx ) noexcept {
        assert( rowIdx >= 0 and rowIdx < nrows(A));
        using idx_t = typename legacyMatrix<T>::idx_t;
        return legacyVector<T,idx_t>( A.n, &A(rowIdx,0), A.ldim );
    }
    
    // Row
    template< typename T >
    inline constexpr auto
    row( const legacyMatrix<T,Layout::RowMajor>& A, typename legacyMatrix<T>::idx_t rowIdx ) noexcept {
        assert( rowIdx >= 0 and rowIdx < nrows(A));
        return legacyVector<T>( A.n, &A(rowIdx,0) );
    }
    
    // Columns
    template< typename T, Layout layout, class SliceSpec >
    inline constexpr auto
    cols( const legacyMatrix<T,layout>& A, SliceSpec&& cols ) noexcept {
        assert( cols.first >= 0 and cols.first < ncols(A));
        assert( cols.second >= 0 and cols.second <= ncols(A));
        return legacyMatrix<T,layout>(
            A.m, cols.second-cols.first,
            &A(0,cols.first), A.ldim
        );
    }
    
    // Column
    template< typename T >
    inline constexpr auto
    col( const legacyMatrix<T>& A, typename legacyMatrix<T>::idx_t colIdx ) noexcept {
        assert( colIdx >= 0 and colIdx < ncols(A));
        return legacyVector<T>( A.m, &A(0,colIdx) );
    }
    
    // Column
    template< typename T >
    inline constexpr auto
    col( const legacyMatrix<T,Layout::RowMajor>& A, typename legacyMatrix<T>::idx_t colIdx ) noexcept {
        assert( colIdx >= 0 and colIdx < ncols(A));
        using idx_t = typename legacyMatrix<T>::idx_t;
        return legacyVector<T,idx_t>( A.m, &A(0,colIdx), A.ldim );
    }

    // Diagonal
    template< typename T, Layout layout >
    inline constexpr auto
    diag( const legacyMatrix<T,layout>& A, int diagIdx = 0 ) noexcept {
        
        using idx_t = typename legacyMatrix<T,layout>::idx_t;
        
        T* ptr  = (diagIdx >= 0) ? &A(0,diagIdx) : &A(-diagIdx,0);
        idx_t n = (diagIdx >= 0)
                    ? std::min( A.m+diagIdx, A.n ) - (idx_t) diagIdx
                    : std::min( A.m, A.n-diagIdx ) + (idx_t) diagIdx;
        
        return legacyVector<T,idx_t>( n, ptr, A.ldim + 1 );
    }

    // slice
    template< typename T, typename int_t, Direction direction, class SliceSpec >
    inline constexpr auto
    slice( const legacyVector<T,int_t,direction>& v, SliceSpec&& rows ) noexcept {
        assert( rows.first >= 0 and rows.first < size(v));
        assert( rows.second >= 0 and rows.second <= size(v));
        return legacyVector<T,int_t,direction>( rows.second-rows.first, &v[rows.first], v.inc );
    }

    // -----------------------------------------------------------------------------
    // Cast to Legacy arrays

    template< typename T, Layout layout >
    inline constexpr auto
    legacy_matrix( legacyMatrix<T,layout>& A ) noexcept { return A; }

    template< typename T, Layout layout >
    inline constexpr auto
    legacy_matrix( const legacyMatrix<T,layout>& A ) noexcept { return A; }

    template< typename T, typename int_t, Direction direction >
    inline constexpr auto
    legacy_vector( legacyVector<T,int_t,direction>& v ) noexcept { return v; }

    template< typename T, typename int_t, Direction direction >
    inline constexpr auto
    legacy_vector( const legacyVector<T,int_t,direction>& v ) noexcept { return v; }

} // namespace tlapack

#endif // __TLAPACK_LEGACYARRAY_HH__
