// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_ARRAY_HH
#define TLAPACK_LEGACY_ARRAY_HH

#include <utility>
#include <type_traits>
#include <cassert>

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/exceptionHandling.hpp"

namespace tlapack {
    
    struct one_t {
        inline constexpr operator int() const { return 1; }
        inline constexpr one_t( int i = 1 ) { assert( i == 1 ); }
    };
    constexpr one_t one = { };

    /** Legacy matrix.
     * 
     * legacyMatrix::ldim is assumed to be positive.
     * 
     * @tparam T Floating-point type
     * @tparam idx_t Index type
     * @tparam L Either Layout::ColMajor or Layout::RowMajor
     */
    template< class T, class idx_t = std::size_t, Layout L = Layout::ColMajor,
        std::enable_if_t< (L == Layout::RowMajor) || (L == Layout::ColMajor), int > = 0
    >
    struct legacyMatrix {
        idx_t m, n;                 ///< Sizes
        T* ptr;                     ///< Pointer to array in memory
        idx_t ldim;                 ///< Leading dimension

        static constexpr Layout layout = L;
        
        inline constexpr T&
        operator()( idx_t i, idx_t j ) const noexcept {
            assert( i >= 0);
            assert( i < m);
            assert( j >= 0);
            assert( j < n);
            return (layout == Layout::ColMajor)
                ? ptr[ i + j*ldim ]
                : ptr[ i*ldim + j ];
        }
        
        inline constexpr legacyMatrix( idx_t m, idx_t n, T* ptr, idx_t ldim )
        : m(m), n(n), ptr(ptr), ldim(ldim)
        {
            tlapack_check_false( m < 0 );
            tlapack_check_false( n < 0 );
            tlapack_check_false( ldim < ((layout == Layout::ColMajor) ? m : n) );
        }
        
        inline constexpr legacyMatrix( idx_t m, idx_t n, T* ptr )
        : m(m), n(n), ptr(ptr), ldim((layout == Layout::ColMajor) ? m : n)
        {
            tlapack_check_false( m < 0 );
            tlapack_check_false( n < 0 );
        }
        
        /**
         * @brief Converts a legacyMatrix<T,idx_t,L> to legacyMatrix<byte>

         * @return constexpr legacyMatrix<byte> Column-major matrix.
         */
        inline constexpr legacyMatrix<byte> in_bytes() const {
            if( L == Layout::ColMajor )
                return legacyMatrix<byte>( m * sizeof(T), n, (byte*) ptr, ldim * sizeof(T) );
            else // if( L == Layout::RowMajor )
                return legacyMatrix<byte>( n * sizeof(T), m, (byte*) ptr, ldim * sizeof(T) );
        }
    };

    /** Legacy vector.
     * 
     * @tparam T Floating-point type
     * @tparam idx_t Index type
     * @tparam D Either Direction::Forward or Direction::Backward
     */
    template< typename T, class idx_t = std::size_t, typename int_t = one_t, Direction D = Direction::Forward >
    struct legacyVector {
        idx_t n;                    ///< Size
        T* ptr;                     ///< Pointer to array in memory
        int_t inc;                  ///< Memory increment

        static constexpr Direction direction = D;
        
        inline constexpr T&
        operator[]( idx_t i ) const noexcept {
            assert( i >= 0);
            assert( i < n);
            return (direction == Direction::Forward)
                ? *(ptr + (i*inc))
                : *(ptr + ((n-1)-i)*inc);
        }
        
        inline constexpr legacyVector( idx_t n, T* ptr, int_t inc = one )
        : n(n), ptr(ptr), inc(inc)
        {
            tlapack_check_false( n < 0 );
            tlapack_check_false( inc == 0 );
        }
    };

    /** Legacy banded matrix.
     * 
     * Mind that the access A(i,j) is valid if, and only if,
     *  max(0,j-ku) <= i <= min(m,j+kl)
     * This class does not perform such a check,
     * otherwise it would lack in performance.
     * 
     * @tparam T Floating-point type
     * @tparam idx_t Index type
     */
    template< typename T, class idx_t = std::size_t >
    struct legacyBandedMatrix {
        idx_t m, n, kl, ku;         ///< Sizes
        T* ptr;                     ///< Pointer to array in memory
        
        /** Access A(i,j) = ptr[ (ku+(i-j)) + j*(ku+kl+1) ]
         * 
         * Mind that this access is valid if, and only if,
         *  max(0,j-ku) <= i <= min(m,j+kl)
         * This operator does not perform such a check,
         * otherwise it would lack in performance.
         * 
         */
        inline constexpr T&
        operator()( idx_t i, idx_t j ) const noexcept {
            assert( i >= 0);
            assert( i < m);
            assert( j >= 0);
            assert( j < n);
            assert( j <= i + ku);
            assert( i <= j + kl);
            return ptr[ (ku+i) + j*(ku+kl) ];
        }
        
        inline constexpr legacyBandedMatrix( idx_t m, idx_t n, idx_t kl, idx_t ku, T* ptr )
        : m(m), n(n), kl(kl), ku(ku), ptr(ptr)
        {
            tlapack_check_false( m < 0 );
            tlapack_check_false( n < 0 );
            tlapack_check_false( (kl + 1 > m && m > 0) || (ku + 1 > n && n > 0) );
        }
    };

    // Workspace
    struct Workspace : public legacyMatrix<byte>
    {
        using idx_t = std::size_t;

        inline constexpr byte* data() const { return ptr; }
        inline constexpr idx_t size() const { return m*n; }

        inline constexpr
        Workspace( byte* ptr = nullptr, idx_t n = 0 )
        : legacyMatrix<byte>( n, 1, ptr ) { }

        inline constexpr
        Workspace( const legacyMatrix<byte>& w ) noexcept
        : legacyMatrix<byte>( w ) { }

        inline constexpr Workspace
        extract( idx_t m, idx_t n ) const {

            assert( m >= 0 && n >= 0 );
            
            if( ldim == this->m || this->n <= 1 ) {
                // contiguous space in memory
                assert( size() >= (m*n) );
                return legacyMatrix<byte>( size()-(m*n), 1, ptr + (m*n) );
            }
            else {
                // non-contiguous space in memory
                assert( this->m >= m && this->n >= n );
                return legacyMatrix<byte>( this->m, this->n - n, ptr + n*ldim, ldim );
            }
        }
    };

    // -----------------------------------------------------------------------------
    // Data traits

    // Layout
    template< typename T, class idx_t, Layout L >
    constexpr Layout layout< legacyMatrix<T,idx_t,L> > = L;

    // Layout
    template< typename T, class idx_t >
    constexpr Layout layout< legacyBandedMatrix<T,idx_t> > = Layout::BandStorage;

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
    struct Create< legacyMatrix<T,idx_t,layout> > {

        using matrix_t = legacyMatrix<T,idx_t,layout>;

        inline constexpr auto
        operator()( std::vector<T>& v, idx_t m, idx_t n ) const {
            v.resize( m*n ); // Allocates space in memory
            return matrix_t( m, n, v.data() );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n, Workspace& rW ) const {
            T* ptr = (T*) W.ptr;
            rW = ( layout == Layout::ColMajor )
                ? W.extract( m*sizeof(T), n )
                : W.extract( n*sizeof(T), m );
            return (rW.ldim == rW.m)
                ? matrix_t( m, n, ptr ) // contiguous space in memory
                : matrix_t( m, n, ptr, rW.ldim/sizeof(T) );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n ) const {
            Workspace rW;
            return operator()( W, m, n, rW );
        }
    };

    template< class T, class idx_t, typename int_t, Direction D >
    struct Create< legacyVector<T,idx_t,int_t,D> > {

        using vector_t = legacyVector<T,idx_t,int_t,D>;

        inline constexpr auto
        operator()( std::vector<T>& v, idx_t m, idx_t n ) const {
            assert( n == 1 );
            v.resize( m ); // Allocates space in memory
            return vector_t( m, v.data() );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n, Workspace& rW ) const {
            assert( n == 1 );
            T* ptr = (T*) W.ptr;
            rW = W.extract( sizeof(T), m );
            return vector_t( m, ptr, (rW.ldim == rW.m) ? 1 : rW.ldim/sizeof(T) );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n ) const {
            Workspace rW;
            return operator()( W, m, n, rW );
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
    legacy_matrix( const legacyVector<T,idx_t,one_t,direction>& v ) noexcept {
        return legacyMatrix<T,idx_t,Layout::ColMajor>( v.n, 1, v.ptr );
    }

    template< typename T, class idx_t, typename int_t, Direction direction >
    inline constexpr auto
    legacy_vector( const legacyVector<T,idx_t,int_t,direction>& v ) noexcept { return v; }

} // namespace tlapack

#endif // TLAPACK_LEGACY_ARRAY_HH
