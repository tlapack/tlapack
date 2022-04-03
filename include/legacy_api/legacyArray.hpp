// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_ARRAY_HH__
#define __TLAPACK_LEGACY_ARRAY_HH__

#include <assert.h>

#include "legacy_api/blas/types.hpp"
#include "blas/exceptionHandling.hpp"

namespace blas {

    /// Runtime direction
    /// TODO: Compatibilize this with Direction from lapack/types
    enum class Direction { Forward = 'F', Backward = 'B' };

    // /// Compile time forward direction
    // struct forward_t {
    //     constexpr operator Direction() const { return Direction::Forward; }
    // };
    // /// Compile time backward direction
    // struct backward_t {
    //     constexpr operator Direction() const { return Direction::Backward; }
    // };

    // // Constant expressions
    // constexpr forward_t forward { };
    // constexpr backward_t backward { };

    /** Legacy matrix.
     * 
     * legacyMatrix::ldim is assumed to be positive.
     * 
     * @tparam T Floating-point type
     * @tparam L Either Layout::ColMajor or Layout::RowMajor
     */
    template< typename T, Layout L = Layout::ColMajor >
    struct legacyMatrix {
        using idx_t = BLAS_SIZE_T;  ///< Index type
        idx_t m, n;                 ///< Sizes
        T* ptr;                     ///< Pointer to array in memory
        idx_t ldim;                 ///< Leading dimension

        static constexpr Layout layout = L;
        
        inline constexpr T&
        operator()( idx_t i, idx_t j ) const noexcept {
            assert( i >= 0);
            assert( i < m);
            assert( j >= 0);
            assert( j < n);
            return (layout == Layout::ColMajor)
                ? ptr[ i + j*ldim ]
                : ptr[ i*ldim + j ];
        }
        
        inline constexpr legacyMatrix( idx_t m, idx_t n, T* ptr, idx_t ldim )
        : m(m), n(n), ptr(ptr), ldim(ldim)
        {
            blas_error_if( m < 0 );
            blas_error_if( n < 0 );
            blas_error_if( ldim < ((layout == Layout::ColMajor) ? m : n) );
        }
    };

    /** Legacy vector.
     * 
     * @tparam T         Floating-point type
     * @tparam direction Either Direction::Forward or Direction::Backward
     */
    template< typename T, typename int_t = one_t, Direction direction = Direction::Forward >
    struct legacyVector {
        using idx_t = BLAS_SIZE_T;  ///< Index type
        idx_t n;                    ///< Size
        T* ptr;                     ///< Pointer to array in memory
        int_t inc;                  ///< Memory increment
        
        inline constexpr T&
        operator[]( idx_t i ) const noexcept {
            assert( i >= 0);
            assert( i < n);
            return (direction == Direction::Forward)
                ? *(ptr + (i*inc))
                : *(ptr + ((n-1)-i)*inc);
        }
        
        inline constexpr legacyVector( idx_t n, T* ptr, int_t inc = one )
        : n(n), ptr(ptr), inc(inc)
        {
            blas_error_if( n < 0 );
            blas_error_if( inc == 0 );
        }
    };

    /** Legacy banded matrix.
     * 
     * Mind that the access A(i,j) is valid if, and only if,
     *  max(0,j-ku) <= i <= min(m,j+kl)
     * This class does not perform such a check,
     * otherwise it would lack in performance.
     * 
     * @tparam T Floating-point type
     */
    template< typename T >
    struct legacyBandedMatrix {
        using idx_t = BLAS_SIZE_T;  ///< Index type
        idx_t m, n, kl, ku;         ///< Sizes
        T* ptr;                     ///< Pointer to array in memory
        
        /** Access A(i,j) = ptr[ (ku+(i-j)) + j*(ku+kl+1) ]
         * 
         * Mind that this access is valid if, and only if,
         *  max(0,j-ku) <= i <= min(m,j+kl)
         * This operator does not perform such a check,
         * otherwise it would lack in performance.
         * 
         */
        inline constexpr T&
        operator()( idx_t i, idx_t j ) const {
            assert( i >= 0);
            assert( i < m);
            assert( j >= 0);
            assert( j < n);
            assert( j <= i + ku);
            assert( i <= j + kl);
            return ptr[ (ku+i) + j*(ku+kl) ];
        }
        
        inline constexpr legacyBandedMatrix( idx_t m, idx_t n, idx_t kl, idx_t ku, T* ptr )
        : m(m), n(n), kl(kl), ku(ku), ptr(ptr)
        {
            blas_error_if( m < 0 );
            blas_error_if( n < 0 );
            blas_error_if( (kl + 1 > m && m > 0) || (ku + 1 > n && n > 0) );
        }
    };

} // namespace blas

namespace lapack {
    using blas::legacyMatrix;
    using blas::legacyVector;
    using blas::legacyBandedMatrix;
}

#endif // __TLAPACK_LEGACY_ARRAY_HH__
