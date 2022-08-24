// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LEGACYARRAY_HH
#define TLAPACK_LEGACY_LEGACYARRAY_HH

#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/base/legacyArray.hpp"

namespace tlapack {

namespace internal {

    template< typename T >
    inline constexpr auto colmajor_matrix(
        T* A, 
        idx_t m, 
        idx_t n, 
        idx_t lda )
    {
        return legacyMatrix<T,idx_t,Layout::ColMajor>{ m, n, A, lda };
    }
    
    template< typename T >
    inline constexpr auto colmajor_matrix(
        T* A, 
        idx_t m, 
        idx_t n )
    {
        return legacyMatrix<T,idx_t,Layout::ColMajor>{ m, n, A, m };
    }

    template< typename T >
    inline constexpr auto rowmajor_matrix(
        T* A, 
        idx_t m, 
        idx_t n, 
        idx_t lda )
    {
        return legacyMatrix<T,idx_t,Layout::RowMajor>{ m, n, A, lda };
    }

    template< typename T >
    inline constexpr auto rowmajor_matrix(
        T* A, 
        idx_t m, 
        idx_t n )
    {
        return legacyMatrix<T,idx_t,Layout::RowMajor>{ m, n, A, n };
    }

    template< typename T >
    inline constexpr auto banded_matrix(
        T* A, 
        idx_t m, 
        idx_t n, 
        idx_t kl, 
        idx_t ku )
    {
        return legacyBandedMatrix<T,idx_t>{ m, n, kl, ku, A };
    }

    template< typename T, typename int_t >
    inline constexpr auto vector( T* x, idx_t n, int_t inc )
    {
        return legacyVector<T,idx_t,int_t>{ n, x, inc };
    }

    template< typename T >
    inline constexpr auto vector( T* x, idx_t n )
    {
        return legacyVector<T,idx_t>{ n, x, one };
    }

    template< typename T, typename int_t >
    inline constexpr auto backward_vector( T* x, idx_t n, int_t inc )
    {
        return legacyVector<T,idx_t,int_t,Direction::Backward>{ n, x, inc };
    }

    template< typename T >
    inline constexpr auto backward_vector( T* x, idx_t n )
    {
        return legacyVector<T,idx_t,one_t,Direction::Backward>{ n, x, one };
    }

} // namespace internal

} // namespace tlapack

#endif // TLAPACK_LEGACY_LEGACYARRAY_HH
