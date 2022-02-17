// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_STDVECTOR_HH__
#define __TLAPACK_STDVECTOR_HH__

#include <vector>
#ifndef TLAPACK_USE_MDSPAN
    #include "legacy_api/legacyArray.hpp"
#else
    #include <experimental/mdspan>
#endif

namespace blas {

    // -----------------------------------------------------------------------------
    // Data traits for std::vector

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

    // Subvector
    template< class T, class Allocator, class SliceSpec >
    inline constexpr auto subvector( const std::vector<T,Allocator>& v, SliceSpec&& rows )
    {
        #ifndef TLAPACK_USE_MDSPAN
            return legacyVector<T>( rows.second - rows.first, (T*) &v[ rows.first ] );
        #else
            return std::experimental::mdspan< T, std::experimental::dextents<1> >(
                (T*) &v[ rows.first ], (std::size_t) (rows.second - rows.first)
            );
        #endif
    }

} // namespace blas

namespace lapack {
    
    using blas::type_t;
    using blas::size_type;

    using blas::size;

    using blas::subvector;

} // namespace lapack

#endif // __TLAPACK_STDVECTOR_HH__
