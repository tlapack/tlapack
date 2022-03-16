// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_MDSPAN_HH__
#define __TLAPACK_MDSPAN_HH__

#include <experimental/mdspan>
#include <type_traits>

#include "blas/arrayTraits.hpp"
#include "legacy_api/legacyArray.hpp"

namespace blas {

    using std::experimental::mdspan;

    // -----------------------------------------------------------------------------
    // Data traits for mdspan

    // Data type
    template< class ET, class Exts, class LP, class AP >
    struct type_trait< mdspan<ET,Exts,LP,AP> > {
        using type = ET;
    };
    // Size type
    template< class ET, class Exts, class LP, class AP >
    struct sizet_trait< mdspan<ET,Exts,LP,AP> > {
        using type = typename mdspan<ET,Exts,LP,AP>::size_type;
    };

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

    // Access policy
    template< class ET, class Exts, class LP, class AP >
    inline constexpr auto
    access_policy( const mdspan<ET,Exts,LP,AP>& x ) {
        /// TODO: Maybe we should get the access type from the layout here?
        return lapack::dense;
    }

    // -----------------------------------------------------------------------------
    // blas functions to access mdspan block operations

    #define isSlice(SliceSpec) std::is_convertible< SliceSpec, std::tuple<std::size_t, std::size_t> >::value

    /** Returns a submatrix from the input matrix
     * 
     * The submatrix uses the same data from the matrix A.
     * 
     * It uses the function `std::experimental::submdspan` from https://github.com/kokkos/mdspan
     * Currently, it only allows for contiguous ranges
     * 
     * @param[in] A             Matrix
     * @param[in] rows          Rows used from A.
     *      mdspan accepts values that can be converted to
     *          - a size_t integer: reprents the index of the row to be used.
     *          - a std::pair(a,b): uses rows from a to b-1
     *          - a std::experimental::full_extent: uses all rows
     * @param[in] cols          Columns used from A.
     *      mdspan accepts values that can be converted to
     *          - a size_t integer: reprents the index of the column to be used.
     *          - a std::pair(a,b): uses columns from a to b-1
     *          - a std::experimental::full_extent: uses all columns
     * 
     * @return mdspan<> submatrix
     */
    template< class ET, class Exts, class LP, class AP,
        class SliceSpecRow, class SliceSpecCol,
        std::enable_if_t< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int > = 0
    >
    inline constexpr auto submatrix(
        const mdspan<ET,Exts,LP,AP>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return std::experimental::submdspan( A, std::forward<SliceSpecRow>(rows), std::forward<SliceSpecCol>(cols) );
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

    // Subvector
    template< class ET, class Exts, class LP, class AP, class SliceSpec,
        std::enable_if_t< isSlice(SliceSpec), int > = 0
    >
    inline constexpr auto subvector( const mdspan<ET,Exts,LP,AP>& v, SliceSpec&& rows ) noexcept
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

        using extents_t = std::experimental::dextents<1>;
        using mapping   = typename layout_stride::template mapping< extents_t >;
        using size_type = typename extents_t::size_type;
        
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

    template< class ET, class Exts, class LP, class AP,
        std::enable_if_t<
        /* Requires: */
            LP::template mapping<Exts>::is_always_strided() &&
            Exts::rank() == 2
        , bool > = true
    >
    inline constexpr auto
    legacy_matrix( const mdspan<ET,Exts,LP,AP>& A ) {
        if( A.stride(0) == 1 )
            return legacyMatrix<ET,Layout::ColMajor>( A.extent(0), A.extent(1), A.data(), A.stride(1) );
        else if( A.stride(1) == 1 )
            return legacyMatrix<ET,Layout::RowMajor>( A.extent(0), A.extent(1), A.data(), A.stride(0) );
        else
            return legacyMatrix<ET>( 0, 0, nullptr, 0 );
    }

} // namespace blas

namespace lapack {
    
    using blas::mdspan;

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

#endif // __TLAPACK_MDSPAN_HH__
