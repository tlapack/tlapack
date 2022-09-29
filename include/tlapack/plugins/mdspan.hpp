// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MDSPAN_HH
#define TLAPACK_MDSPAN_HH

#include <experimental/mdspan>

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/legacyArray.hpp"

namespace tlapack {

    using std::experimental::mdspan;

    // -----------------------------------------------------------------------------
    // blas functions to access mdspan properties

    // Size
    template< class ET, class Exts, class LP, class AP >
    inline constexpr auto
    size( const mdspan<ET,Exts,LP,AP>& x ) {
        return x.size();
    }
    // Number of rows
    template< class ET, class Exts, class LP, class AP >
    inline constexpr auto
    nrows( const mdspan<ET,Exts,LP,AP>& x ) {
        return x.extent(0);
    }
    // Number of columns
    template< class ET, class Exts, class LP, class AP >
    inline constexpr auto
    ncols( const mdspan<ET,Exts,LP,AP>& x ) {
        return x.extent(1);
    }

    // Read policy
    template< class ET, class Exts, class LP, class AP >
    inline constexpr auto
    read_policy( const mdspan<ET,Exts,LP,AP>& x ) {
        /// TODO: Maybe we should get the access type from the layout here?
        return dense;
    }

    // Write policy
    template< class ET, class Exts, class LP, class AP >
    inline constexpr auto
    write_policy( const mdspan<ET,Exts,LP,AP>& x ) {
        /// TODO: Maybe we should get the access type from the layout here?
        return dense;
    }

    // -----------------------------------------------------------------------------
    // blas functions to access mdspan block operations

    #define isSlice(SliceSpec) std::is_convertible< SliceSpec, std::tuple<std::size_t, std::size_t> >::value

    // Slice
    template< class ET, class Exts, class LP, class AP,
        class SliceSpecRow, class SliceSpecCol,
        std::enable_if_t< isSlice(SliceSpecRow) || isSlice(SliceSpecCol), int > = 0
    >
    inline constexpr auto slice(
        const mdspan<ET,Exts,LP,AP>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return std::experimental::submdspan( A, std::forward<SliceSpecRow>(rows), std::forward<SliceSpecCol>(cols) );
    }

    // Slice
    template< class ET, class Exts, class LP, class AP,
        class SliceSpec,
        std::enable_if_t< isSlice(SliceSpec) && (Exts::rank() == 2), int > = 0
    >
    inline constexpr auto slice( const mdspan<ET,Exts,LP,AP>& A, SliceSpec&& rows ) noexcept
    {
        return std::experimental::submdspan( A, std::forward<SliceSpec>(rows), 0 );
    }

    // Rows
    template< class ET, class Exts, class LP, class AP, class SliceSpec,
        std::enable_if_t< isSlice(SliceSpec), int > = 0
    >
    inline constexpr auto rows( const mdspan<ET,Exts,LP,AP>& A, SliceSpec&& rows ) noexcept
    {
        return std::experimental::submdspan( A, std::forward<SliceSpec>(rows), std::experimental::full_extent );
    }

    // Row
    template< class ET, class Exts, class LP, class AP >
    inline constexpr auto row( const mdspan<ET,Exts,LP,AP>& A, std::size_t rowIdx ) noexcept
    {
        return std::experimental::submdspan( A, rowIdx, std::experimental::full_extent );
    }

    // Columns
    template< class ET, class Exts, class LP, class AP, class SliceSpec,
        std::enable_if_t< isSlice(SliceSpec), int > = 0
    >
    inline constexpr auto cols( const mdspan<ET,Exts,LP,AP>& A, SliceSpec&& cols ) noexcept
    {
        return std::experimental::submdspan( A, std::experimental::full_extent, std::forward<SliceSpec>(cols) );
    }

    // Column
    template< class ET, class Exts, class LP, class AP >
    inline constexpr auto col( const mdspan<ET,Exts,LP,AP>& A, std::size_t colIdx ) noexcept
    {
        return std::experimental::submdspan( A, std::experimental::full_extent, colIdx );
    }

    // Slice
    template< class ET, class Exts, class LP, class AP, class SliceSpec,
        std::enable_if_t< isSlice(SliceSpec) && (Exts::rank() == 1), int > = 0
    >
    inline constexpr auto slice( const mdspan<ET,Exts,LP,AP>& v, SliceSpec&& rows ) noexcept
    {
        return std::experimental::submdspan( v, std::forward<SliceSpec>(rows) );
    }

    // Extract a diagonal from a matrix
    template< class ET, class Exts, class LP, class AP, class int_t,
        std::enable_if_t<
        /* Requires: */
            LP::template mapping<Exts>::is_always_strided()
        , bool > = true
    >
    inline constexpr auto diag(
        const mdspan<ET,Exts,LP,AP>& A,
        int_t diagIdx = 0 )
    {
        using std::abs;
        using std::min;
        using std::array;
        using std::experimental::layout_stride;

        using size_type = typename mdspan<ET,Exts,LP,AP>::size_type;
        using extents_t = std::experimental::dextents<size_type,1>;
        using mapping   = typename layout_stride::template mapping< extents_t >;
        
        // mdspan components
        auto ptr = A.accessor().offset( A.data(),
            (diagIdx >= 0)
                ? A.mapping()(0,diagIdx)
                : A.mapping()(-diagIdx,0)
        );
        auto map = mapping(
            extents_t( (diagIdx >= 0)
                ? min( A.extent(0)+diagIdx, A.extent(1) ) - (size_type) diagIdx
                : min( A.extent(0), A.extent(1)-diagIdx ) + (size_type) diagIdx
            ),
            array<size_type, 1>{ A.stride(0) + A.stride(1) }
        );
        auto acc_pol = typename AP::offset_policy(A.accessor());

        // return
        return mdspan< ET, extents_t, layout_stride, AP > (
            std::move(ptr), std::move(map), std::move(acc_pol)
        );
    }

    #undef isSlice

    // -----------------------------------------------------------------------------
    // Create objects

    template< class ET, class Exts, class LP, class AP >
    struct Create< mdspan<ET,Exts,LP,AP> >
    {
        using idx_t = typename mdspan<ET,Exts,LP,AP>::size_type;
        using extents_t = std::experimental::dextents<idx_t,Exts::rank()>;

        template<std::enable_if_t< Exts::rank() == 1 ,int> =0>
        inline constexpr auto
        operator()( ET* ptr, idx_t m, idx_t n ) const {
                return mdspan<ET,extents_t,LP,AP>( ptr, m );
        }

        template<std::enable_if_t< Exts::rank() == 2 ,int> =0>
        inline constexpr auto
        operator()( ET* ptr, idx_t m, idx_t n ) const {
                return mdspan<ET,extents_t,LP,AP>( ptr, m, n );
        }

        template<std::enable_if_t< Exts::rank() == 2 ,int> =0>
        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n, Workspace& rW ) const
        {
            using std::array;
            using std::experimental::layout_stride;
            using mapping   = typename layout_stride::template mapping< extents_t >;
            using matrix_t  = mdspan<ET,extents_t,layout_stride,AP>;

            using std::experimental::layout_right; // Row-Major
            using std::experimental::layout_left; // Col-Major

            assert( m >= 0 && n >= 0 );
            
            // Variables to be forwarded to the returned matrix
            ET* ptr = (ET*) W.ptr;
            mapping map;

            if( W.ldim == m*sizeof(ET) || W.n <= 1 )
            {
                // contiguous space in memory
                const std::size_t nBytes = m*n*sizeof(ET);
                assert( W.size() >= nBytes );
                rW = legacyMatrix<byte>( W.size()-nBytes, 1, W.ptr + nBytes );

                map = mapping( extents_t(m,n), array<idx_t,2>{m,1} );
            }
            else
            {
                // non-contiguous space in memory

                if( W.m >= m*sizeof(ET) && W.n >= n )
                {
                    rW = legacyMatrix<byte>( W.m, W.n-n, W.ptr+n*W.ldim, W.ldim );

                    map = mapping( extents_t(m,n), array<idx_t,2>{W.ldim/sizeof(ET),1} );
                }
                else
                {
                    assert( W.m >= n*sizeof(ET) && W.n >= m );
                    rW = legacyMatrix<byte>( W.m, W.n-m, W.ptr+m*W.ldim, W.ldim );

                    map = mapping( extents_t(m,n), array<idx_t,2>{1,W.ldim/sizeof(ET)} );
                }
            }

            return matrix_t( std::move(ptr), std::move(map) );
        }

        template<std::enable_if_t< Exts::rank() == 1 ,int> =0>
        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n, Workspace& rW ) const
        {
            using std::array;
            using std::experimental::layout_stride;
            using mapping   = typename layout_stride::template mapping< extents_t >;
            using matrix_t  = mdspan<ET,extents_t,layout_stride,AP>;

            using std::experimental::layout_right; // Row-Major
            using std::experimental::layout_left; // Col-Major

            assert( m >= 0 && n == 1 );
            
            // Variables to be forwarded to the returned matrix
            ET* ptr = (ET*) W.ptr;
            mapping map;

            if( W.ldim == m*sizeof(ET) || W.n <= 1 )
            {
                // contiguous space in memory
                const std::size_t nBytes = m*sizeof(ET);
                assert( W.size() >= nBytes );
                rW = legacyMatrix<byte>( W.size()-nBytes, 1, W.ptr + nBytes );

                map = mapping( extents_t(m), array<idx_t,1>{1} );
            }
            else
            {
                // non-contiguous space in memory
                assert( W.m >= sizeof(ET) && W.n >= m );
                rW = legacyMatrix<byte>( W.m, W.n-m, W.ptr+m*W.ldim, W.ldim );

                map = mapping( extents_t(m), array<idx_t,1>{W.ldim/sizeof(ET)} );
            }

            return matrix_t( std::move(ptr), std::move(map) );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n ) const {
            Workspace rW;
            return operator()( W, m, n, rW );
        }
    };

    // -----------------------------------------------------------------------------
    // Convert to legacy array

    // /// alias has_blas_type for arrays
    // template< class ET, class Exts, class LP, class AP,
    //     std::enable_if_t<
    //     /* Requires: */
    //         has_blas_type_v<ET> &&
    //         Exts::rank() <= 2 &&
    //         LP::template mapping<Exts>::is_always_strided()
    //     , bool > = true
    // >
    // struct has_blas_type< mdspan<ET,Exts,LP,AP> > {
    //     using type = ET;
    //     static constexpr bool value =
    //         has_blas_type_v<type> && ;
    // };

    // template< class ET, class Exts, class LP, class AP,
    //     std::enable_if_t<
    //     /* Requires: */
    //         LP::template mapping<Exts>::is_always_strided() &&
    //         Exts::rank() == 2
    //     , bool > = true
    // >
    // inline constexpr auto
    // legacy_matrix( const mdspan<ET,Exts,LP,AP>& A ) {
    //     if( A.stride(0) == 1 )
    //         return legacyMatrix<ET,Layout::ColMajor>( A.extent(0), A.extent(1), A.data(), A.stride(1) );
    //     else if( A.stride(1) == 1 )
    //         return legacyMatrix<ET,Layout::RowMajor>( A.extent(0), A.extent(1), A.data(), A.stride(0) );
    //     else
    //         return legacyMatrix<ET>( 0, 0, nullptr, 0 );
    // }

} // namespace tlapack

#endif // TLAPACK_MDSPAN_HH
