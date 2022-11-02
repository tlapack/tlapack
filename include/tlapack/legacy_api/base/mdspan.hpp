// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_MDSPAN_HH
#define TLAPACK_LEGACY_MDSPAN_HH

#include <experimental/mdspan> // Use mdspan for multidimensional arrays

namespace tlapack {

using std::experimental::mdspan;

namespace internal {

    // -----------------------------------------------------------------------------
    /** Returns a Matrix object representing a column major matrix
     * 
     * @param A                 serial data
     * @param m                 number of rows
     * @param n                 number of columns
     * @param lda               leading dimension 
     * 
     * @return mdspan< T, dextents<2>, layout_stride > 
     *      matrix object using the abstraction A(i,j) = i + j * lda
     */
    template< typename T, typename integral_type >
    inline constexpr auto colmajor_matrix(
        T* A, 
        std::experimental::dextents<2>::size_type m, 
        std::experimental::dextents<2>::size_type n, 
        integral_type lda ) noexcept
    {
        using std::array;
        using std::experimental::dextents;
        using extents_t = dextents<2>;
        using std::experimental::layout_stride;
        using mapping = typename layout_stride::template mapping< extents_t >;

        return mdspan< T, extents_t, layout_stride > (
            A, mapping( extents_t(m,n), array<integral_type, 2>{1,lda} )
        );
    }
    
    template< typename T >
    inline constexpr auto colmajor_matrix(
        T* A, 
        std::experimental::dextents<2>::size_type m, 
        std::experimental::dextents<2>::size_type n ) noexcept
    {
        using std::experimental::dextents;
        using extents_t = dextents<2>;
        using std::experimental::layout_left;
        using mapping = typename layout_left::template mapping< extents_t >;

        return mdspan< T, extents_t, layout_left > (
            A, mapping( extents_t(m,n) )
        );
    }
    
    template< typename T, typename integral_type >
    inline constexpr auto rowmajor_matrix(
        T* A, 
        std::experimental::dextents<2>::size_type m, 
        std::experimental::dextents<2>::size_type n, 
        integral_type lda ) noexcept
    {
        using std::array;
        using std::experimental::dextents;
        using extents_t = dextents<2>;
        using std::experimental::layout_stride;
        using mapping = typename layout_stride::template mapping< extents_t >;

        return mdspan< T, extents_t, layout_stride > (
            A, mapping( extents_t(m,n), array<integral_type, 2>{lda,1} )
        );
    }
    
    template< typename T >
    inline constexpr auto rowmajor_matrix(
        T* A, 
        std::experimental::dextents<2>::size_type m, 
        std::experimental::dextents<2>::size_type n ) noexcept
    {
        using std::experimental::dextents;
        using extents_t = dextents<2>;
        using std::experimental::layout_right;
        using mapping = typename layout_right::template mapping< extents_t >;

        return mdspan< T, extents_t, layout_right > (
            A, mapping( extents_t(m,n) )
        );
    }

    template< typename T, typename integral_type >
    inline constexpr auto vector(
        T* x,
        std::experimental::dextents<1>::size_type n,
        integral_type ldim ) noexcept
    {
        using std::array;
        using std::experimental::dextents;
        using extents_t = dextents<1>;
        using std::experimental::layout_stride;
        using mapping = typename layout_stride::template mapping< extents_t >;

        return mdspan< T, extents_t, layout_stride > (
            x, mapping( extents_t(n), array<integral_type, 1>{ldim} )
        );
    }

    template< typename T >
    inline constexpr auto vector(
        T* x,
        std::experimental::dextents<1>::size_type n ) noexcept
    {
        using std::experimental::dextents;
        using extents_t = dextents<1>;
        using std::experimental::layout_left;
        using mapping = typename layout_left::template mapping< extents_t >;

        return mdspan< T, extents_t, layout_left > (
            x, mapping( extents_t(n) )
        );
    }

    // Transpose
    template< class ET, class Exts, class AP >
    inline constexpr auto transpose(
        const mdspan<ET,Exts,std::experimental::layout_stride,AP>& A ) noexcept
    {
        using std::array;
        using std::experimental::layout_stride;
        using mapping = typename layout_stride::template mapping< Exts >;
        using size_type = typename Exts::size_type;
        
        // constants
        const size_type m  = A.extent(0);
        const size_type n  = A.extent(1);
        const size_type s0 = A.stride(0);
        const size_type s1 = A.stride(1);

        // return
        return mdspan<ET,Exts,layout_stride,AP>(
            A.data(),
            mapping(
                Exts(n,m),
                array<size_type, 2>{s1,s0}
            ),
            typename AP::offset_policy(A.accessor())
        );
    }

    // Transpose
    template< class ET, class Exts, class AP >
    inline constexpr auto transpose(
        const mdspan<ET,Exts,std::experimental::layout_left,AP>& A ) noexcept
    {
        using std::experimental::layout_right;
        using mapping = typename layout_right::template mapping< Exts >;
        using size_type = typename Exts::size_type;
        
        // constants
        const size_type m  = A.extent(0);
        const size_type n  = A.extent(1);

        // return
        return mdspan<ET,Exts,layout_right,AP>(
            A.data(),
            mapping( Exts(n,m) ),
            typename AP::offset_policy(A.accessor())
        );
    }

    // Transpose
    template< class ET, class Exts, class AP >
    inline constexpr auto transpose(
        const mdspan<ET,Exts,std::experimental::layout_right,AP>& A ) noexcept
    {
        using std::experimental::layout_left;
        using mapping = typename layout_left::template mapping< Exts >;
        using size_type = typename Exts::size_type;
        
        // constants
        const size_type m  = A.extent(0);
        const size_type n  = A.extent(1);

        // return
        return mdspan<ET,Exts,layout_left,AP>(
            A.data(),
            mapping( Exts(n,m) ),
            typename AP::offset_policy(A.accessor())
        );
    }

} // namespace internal

} // namespace tlapack

#endif // TLAPACK_LEGACY_MDSPAN_HH
