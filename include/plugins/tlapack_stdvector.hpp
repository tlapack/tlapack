// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_STDVECTOR_HH__
#define __TLAPACK_STDVECTOR_HH__

#include <vector>
#include "blas/arrayTraits.hpp"

#ifndef TLAPACK_USE_MDSPAN
    #include "legacy_api/legacyArray.hpp"
#else
    #include <experimental/mdspan>
#endif

namespace blas {

    // -----------------------------------------------------------------------------
    // Data traits for std::vector

    // Data type
    template< class T, class Allocator >
    struct type_trait< std::vector<T,Allocator> > {
        using type = T;
    };
    // Size type
    template< class T, class Allocator >
    struct sizet_trait< std::vector<T,Allocator> > {
        using type = typename std::vector<T,Allocator>::size_type;
    };

    // -----------------------------------------------------------------------------
    // blas functions to access std::vector properties

    // Size
    template< class T, class Allocator >
    inline constexpr auto
    size( const std::vector<T,Allocator>& x ) {
        return x.size();
    }

    // -----------------------------------------------------------------------------
    // blas functions to access std::vector block operations

    // slice
    template< class T, class Allocator, class SliceSpec >
    inline constexpr auto slice( const std::vector<T,Allocator>& v, SliceSpec&& rows )
    {
        #ifndef TLAPACK_USE_MDSPAN
            return legacyVector<T>( rows.second - rows.first, (T*) &v[ rows.first ] );
        #else
            return std::experimental::mdspan< T, std::experimental::dextents<1> >(
                (T*) &v[ rows.first ], (std::size_t) (rows.second - rows.first)
            );
        #endif
    }

    // interpretAsMatrix
    template< class T, class Allocator >
    inline constexpr
    auto
    interpretAsMatrix( const std::vector<T,Allocator>& v ) noexcept
    {
        #ifndef TLAPACK_USE_MDSPAN
            return legacyMatrix< T >( v.size(), 1, &v[0], v.size() );
        #else
            using extents_t = std::experimental::extents<std::experimental::dynamic_extent,1>;
            return std::experimental::mdspan< T, extents_t >( &v[0], extents_t(v.size()) );
        #endif
    }

} // namespace blas

namespace lapack {

    using blas::size;
    using blas::slice;
    using blas::interpretAsMatrix;

} // namespace lapack

#endif // __TLAPACK_STDVECTOR_HH__
