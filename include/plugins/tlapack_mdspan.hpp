// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_MDSPAN_HH__
#define __TLAPACK_MDSPAN_HH__

#include <experimental/mdspan>
#include <type_traits>

namespace blas {

    using std::experimental::mdspan;

    // -----------------------------------------------------------------------------
    // enable_if_t is defined in C++14; here's a C++11 definition
    #if __cplusplus >= 201402L
        using std::enable_if_t;
    #else
        template< bool B, class T = void >
        using enable_if_t = typename enable_if<B,T>::type;
    #endif

    // -----------------------------------------------------------------------------
    // is_convertible_v is defined in C++17; here's a C++11 definition
    #if __cplusplus >= 201703L
        using std::is_convertible_v;
    #else
        template< class From, class To >
        constexpr bool is_convertible_v = std::is_convertible<From, To>::value;
    #endif

    // -----------------------------------------------------------------------------
    // Data traits for mdspan

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

    // -----------------------------------------------------------------------------
    // blas functions to access mdspan block operations

    #define isSlice(SliceSpec) is_convertible_v< SliceSpec, std::tuple<std::size_t, std::size_t> >

    // Submatrix
    template< class ET, class Exts, class LP, class AP,
        class SliceSpecRow, class SliceSpecCol,
        enable_if_t< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int > = 0
    >
    inline constexpr auto submatrix(
        const mdspan<ET,Exts,LP,AP>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return std::experimental::submdspan( A, std::forward<SliceSpecRow>(rows), std::forward<SliceSpecCol>(cols) );
    }

    // Rows
    template< class ET, class Exts, class LP, class AP, class SliceSpec,
        enable_if_t< isSlice(SliceSpec), int > = 0
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
        enable_if_t< isSlice(SliceSpec), int > = 0
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
        enable_if_t< isSlice(SliceSpec), int > = 0
    >
    inline constexpr auto subvector( const mdspan<ET,Exts,LP,AP>& v, SliceSpec&& rows ) noexcept
    {
        return std::experimental::submdspan( v, std::forward<SliceSpec>(rows) );
    }

    // Extract a diagonal from a matrix
    template< class ET, class Exts, class LP, class AP,
        enable_if_t<
        /* Requires: */
            LP::template mapping<Exts>::is_always_strided()
        , bool > = true
    >
    inline constexpr auto diag(
        const mdspan<ET,Exts,LP,AP>& A,
        int diagIdx = 0 )
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
            extents_t( min( A.extent(0), A.extent(1) ) - abs(diagIdx) ),
            array<size_type, 1>{ A.stride(0) + A.stride(1) }
        );
        auto acc_pol = typename AP::offset_policy(A.accessor());

        // return
        return mdspan< ET, extents_t, layout_stride, AP > (
            std::move(ptr), std::move(map), std::move(acc_pol)
        );
    }

    #undef isSlice

} // namespace blas

namespace lapack {
    
    using blas::mdspan;
    using blas::type_trait;
    using blas::sizet_trait;

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
