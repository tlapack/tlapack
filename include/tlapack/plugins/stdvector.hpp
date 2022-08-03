// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STDVECTOR_HH
#define TLAPACK_STDVECTOR_HH

#include <vector>
#include "tlapack/base/arrayTraits.hpp"

#ifndef TLAPACK_USE_MDSPAN
    #include "tlapack/legacy_api/legacyArray.hpp"
#else
    #include <experimental/mdspan>
#endif

namespace tlapack {

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

} // namespace tlapack

#endif // TLAPACK_STDVECTOR_HH
