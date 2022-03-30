// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACYARRAY_HH__
#define __TLAPACK_LEGACYARRAY_HH__

#include <utility>

#include "legacy_api/legacyArray.hpp"
#include "blas/arrayTraits.hpp"

namespace blas {

    // -----------------------------------------------------------------------------
    // Data traits

    // Data type
    template< typename T, Layout layout >
    struct type_trait< legacyMatrix<T,layout> > { using type = T; };
    // Size type
    template< typename T, Layout layout >
    struct sizet_trait< legacyMatrix<T,layout> > { using type = typename legacyMatrix<T>::idx_t; };
    // Layout type
    template< typename T >
    struct layout_trait< legacyMatrix<T> > { using type = ColMajor_t; };
    template< typename T >
    struct layout_trait< legacyMatrix<T,Layout::RowMajor> > { using type = RowMajor_t; };

    /// Specialization of has_blas_type for arrays.
    template< typename T, Layout L >
    struct allow_optblas< legacyMatrix<T,L> > {
        using type = T;
        static constexpr bool value = allow_optblas_v<type>;
    };

    // Data type
    template< typename T, typename int_t, Direction direction >
    struct type_trait< legacyVector<T,int_t,direction> > { using type = T; };
    // Size type
    template< typename T, typename int_t, Direction direction >
    struct sizet_trait< legacyVector<T,int_t,direction> > { using type = typename legacyVector<T>::idx_t; };
    
    /// Specialization of has_blas_type for arrays.
    template< typename T, typename int_t, Direction direction >
    struct allow_optblas< legacyVector<T,int_t,direction> > {
        using type = T;
        static constexpr bool value = allow_optblas_v<type>;
    };

    // Data type
    template< typename T >
    struct type_trait< legacyBandedMatrix<T> > { using type = T; };
    // Size type
    template< typename T >
    struct sizet_trait< legacyBandedMatrix<T> > { using type = typename legacyBandedMatrix<T>::idx_t; };
    // Layout type
    template< typename T >
    struct layout_trait< legacyBandedMatrix<T> > { using type = Banded_t; };

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
        return lapack::dense;
    }

    // Write policy
    template< typename T, Layout layout >
    inline constexpr auto
    write_policy( const legacyMatrix<T,layout>& A ) {
        return lapack::dense;
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
        return lapack::band_t {
            (std::size_t) A.kl, (std::size_t) A.ku
        };
    }

    // Access policy
    template< typename T >
    inline constexpr auto
    write_policy( const legacyBandedMatrix<T>& A ) {
        return lapack::band_t {
            (std::size_t) A.kl, (std::size_t) A.ku
        };
    }

    // -----------------------------------------------------------------------------
    // Data blocks
    
    // Submatrix
    template< typename T, Layout layout, class SliceSpecRow, class SliceSpecCol >
    inline constexpr auto
    submatrix( const legacyMatrix<T,layout>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept {
        return legacyMatrix<T,layout>(
            rows.second-rows.first, cols.second-cols.first,
            &A(rows.first,cols.first), A.ldim
        );
    }
    
    // Rows
    template< typename T, Layout layout, class SliceSpec >
    inline constexpr auto
    rows( const legacyMatrix<T,layout>& A, SliceSpec&& rows ) noexcept {
        return legacyMatrix<T,layout>(
            rows.second-rows.first, A.n,
            &A(rows.first,0), A.ldim
        );
    }
    
    // Row
    template< typename T >
    inline constexpr auto
    row( const legacyMatrix<T>& A, typename legacyMatrix<T>::idx_t rowIdx ) noexcept {
        using idx_t = typename legacyMatrix<T>::idx_t;
        return legacyVector<T,idx_t>( A.n, &A(rowIdx,0), A.ldim );
    }
    
    // Row
    template< typename T >
    inline constexpr auto
    row( const legacyMatrix<T,Layout::RowMajor>& A, typename legacyMatrix<T>::idx_t rowIdx ) noexcept {
        return legacyVector<T>( A.n, &A(rowIdx,0) );
    }
    
    // Columns
    template< typename T, Layout layout, class SliceSpec >
    inline constexpr auto
    cols( const legacyMatrix<T,layout>& A, SliceSpec&& cols ) noexcept {
        return legacyMatrix<T,layout>(
            A.m, cols.second-cols.first,
            &A(0,cols.first), A.ldim
        );
    }
    
    // Column
    template< typename T >
    inline constexpr auto
    col( const legacyMatrix<T>& A, typename legacyMatrix<T>::idx_t colIdx ) noexcept {
        return legacyVector<T>( A.m, &A(0,colIdx) );
    }
    
    // Column
    template< typename T >
    inline constexpr auto
    col( const legacyMatrix<T,Layout::RowMajor>& A, typename legacyMatrix<T>::idx_t colIdx ) noexcept {
        using idx_t = typename legacyMatrix<T>::idx_t;
        return legacyVector<T,idx_t>( A.m, &A(0,colIdx), A.ldim );
    }

    // Diagonal
    template< typename T, Layout layout, class int_t >
    inline constexpr auto
    diag( const legacyMatrix<T,layout>& A, int_t diagIdx = 0 ) noexcept {
        
        using idx_t = typename legacyMatrix<T,layout>::idx_t;
        
        T* ptr  = (diagIdx >= 0) ? &A(0,diagIdx) : &A(-diagIdx,0);
        idx_t n = (diagIdx >= 0)
                    ? std::min( A.m+diagIdx, A.n ) - (idx_t) diagIdx
                    : std::min( A.m, A.n-diagIdx ) + (idx_t) diagIdx;
        
        return legacyVector<T,idx_t>( n, ptr, A.ldim + 1 );
    }

    // Subvector
    template< typename T, typename int_t, Direction direction, class SliceSpec >
    inline constexpr auto
    subvector( const legacyVector<T,int_t,direction>& v, SliceSpec&& rows ) noexcept {
        return legacyVector<T,int_t,direction>( rows.second-rows.first, &v[rows.first], v.inc );
    }

    // interpretAsMatrix
    template< typename T, Layout layout >
    inline constexpr
    const auto&
    interpretAsMatrix( const legacyMatrix<T,layout>& A ) noexcept
    {
        return A;
    }

    // interpretAsMatrix
    template< typename T, typename int_t, Direction direction >
    inline constexpr
    auto
    interpretAsMatrix( const legacyVector<T,int_t,direction>& v )
    {
        if( v.inc == 1 )
            return legacyMatrix< T >( v.n, 1, v.ptr, v.n );
        else
            throw Error("Invalid conversion from legacyVector to legacyMatrix");
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

} // namespace blas

namespace lapack {

    using blas::size;
    using blas::nrows;
    using blas::ncols;
    using blas::read_policy;
    using blas::write_policy;

    using blas::submatrix;
    using blas::rows;
    using blas::row;
    using blas::cols;
    using blas::col;
    using blas::diag;

    using blas::subvector;

    using blas::interpretAsMatrix;

} // namespace lapack

#endif // __TLAPACK_LEGACYARRAY_HH__
