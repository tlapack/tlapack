// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_ARRAY_TRAITS__
#define __TBLAS_ARRAY_TRAITS__

#include <type_traits>

namespace blas {

    /// Data type
    template< class T > struct type_trait {};
    template< class T >
    using type_t = typename type_trait< T >::type;
    
    /// Size type
    template< class T > struct sizet_trait {};
    template< class T >
    using size_type = typename sizet_trait< T >::type;

    /// Runtime layouts
    enum class Layout {
        ColMajor = 'C',
        RowMajor = 'R', 
        Scalar = 0, 
        StridedVector = 1
    };

    /// Verifies if the set of data structures allow optimization using optBLAS.
    template<class...>
    struct allow_optblas {
        using type = bool; ///< Floating datatype.
        static constexpr Layout layout = Layout::Scalar; ///< Layout type.
        static constexpr bool value = false; ///< True if it allows optBLAS.
    };

    /// Alias for allow_optblas<>::type.
    template<class... Ts>
    using allow_optblas_t = typename allow_optblas< Ts... >::type;

    /// Alias for allow_optblas<>::layout.
    template<class... Ts>
    constexpr Layout allow_optblas_l = allow_optblas< Ts... >::layout;

    /// Alias for allow_optblas<>::value.
    template<class... Ts>
    constexpr bool allow_optblas_v = allow_optblas< Ts... >::value;
}

#endif // __TBLAS_ARRAY_TRAITS__