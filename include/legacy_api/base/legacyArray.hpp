// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_LEGACYARRAY_HH__
#define __TLAPACK_LEGACY_LEGACYARRAY_HH__

#include "legacy_api/legacyArray.hpp"

namespace tlapack {

namespace internal {

    template< typename T >
    inline constexpr auto colmajor_matrix(
        T* A, 
        TLAPACK_SIZE_T m, 
        TLAPACK_SIZE_T n, 
        TLAPACK_SIZE_T lda )
    {
        return legacyMatrix<T,Layout::ColMajor>{ m, n, A, lda };
    }
    
    template< typename T >
    inline constexpr auto colmajor_matrix(
        T* A, 
        TLAPACK_SIZE_T m, 
        TLAPACK_SIZE_T n )
    {
        return legacyMatrix<T,Layout::ColMajor>{ m, n, A, m };
    }

    template< typename T >
    inline constexpr auto rowmajor_matrix(
        T* A, 
        TLAPACK_SIZE_T m, 
        TLAPACK_SIZE_T n, 
        TLAPACK_SIZE_T lda )
    {
        return legacyMatrix<T,Layout::RowMajor>{ m, n, A, lda };
    }

    template< typename T >
    inline constexpr auto rowmajor_matrix(
        T* A, 
        TLAPACK_SIZE_T m, 
        TLAPACK_SIZE_T n )
    {
        return legacyMatrix<T,Layout::RowMajor>{ m, n, A, n };
    }

    template< typename T >
    inline constexpr auto banded_matrix(
        T* A, 
        TLAPACK_SIZE_T m, 
        TLAPACK_SIZE_T n, 
        TLAPACK_SIZE_T kl, 
        TLAPACK_SIZE_T ku )
    {
        return legacyBandedMatrix<T>{ m, n, kl, ku, A };
    }

    template< typename T, typename int_t >
    inline constexpr auto vector( T* x, TLAPACK_SIZE_T n, int_t inc )
    {
        return legacyVector<T,int_t>{ n, x, inc };
    }

    template< typename T >
    inline constexpr auto vector( T* x, TLAPACK_SIZE_T n )
    {
        return legacyVector<T>{ n, x, one };
    }

    template< typename T, typename int_t >
    inline constexpr auto backward_vector( T* x, TLAPACK_SIZE_T n, int_t inc )
    {
        return legacyVector<T,int_t,Direction::Backward>{ n, x, inc };
    }

    template< typename T >
    inline constexpr auto backward_vector( T* x, TLAPACK_SIZE_T n )
    {
        return legacyVector<T,one_t,Direction::Backward>{ n, x, one };
    }

    // Transpose
    template< typename T >
    inline constexpr auto transpose(
        const legacyMatrix<T,Layout::ColMajor>& A )
    {
        return legacyMatrix<T,Layout::RowMajor>{ A.n, A.m, A.ptr, A.ldim };
    }

    // Transpose
    template< typename T >
    inline constexpr auto transpose(
        const legacyMatrix<T,Layout::RowMajor>& A )
    {
        return legacyMatrix<T,Layout::ColMajor>{ A.n, A.m, A.ptr, A.ldim };
    }

} // namespace internal

} // namespace tlapack

#endif // __TLAPACK_LEGACY_LEGACYARRAY_HH__
