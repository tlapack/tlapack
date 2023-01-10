/// @file legacyArray.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_ARRAY_HH
#define TLAPACK_LEGACY_ARRAY_HH

#include <cassert>
#include "tlapack/base/types.hpp"
#include "tlapack/base/exceptionHandling.hpp"

namespace tlapack {

    /** Legacy matrix.
     * 
     * legacyMatrix::ldim is assumed to be positive.
     * 
     * @tparam T Floating-point type
     * @tparam idx_t Index type
     * @tparam L Either Layout::ColMajor or Layout::RowMajor
     */
    template< class T, class idx_t = std::size_t, Layout L = Layout::ColMajor,
        std::enable_if_t< (L == Layout::RowMajor) || (L == Layout::ColMajor), int > = 0
    >
    struct legacyMatrix {
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
            tlapack_check( m >= 0 );
            tlapack_check( n >= 0 );
            tlapack_check( ldim >= ((layout == Layout::ColMajor) ? m : n) );
        }
        
        inline constexpr legacyMatrix( idx_t m, idx_t n, T* ptr )
        : m(m), n(n), ptr(ptr), ldim((layout == Layout::ColMajor) ? m : n)
        {
            tlapack_check( m >= 0 );
            tlapack_check( n >= 0 );
        }
        
        /**
         * @brief Converts a legacyMatrix<T,idx_t,L> to legacyMatrix<byte>

         * @return constexpr legacyMatrix<byte> Column-major matrix.
         */
        inline constexpr legacyMatrix<byte> in_bytes() const {
            if( L == Layout::ColMajor )
                return legacyMatrix<byte>( m * sizeof(T), n, (byte*) ptr, ldim * sizeof(T) );
            else // if( L == Layout::RowMajor )
                return legacyMatrix<byte>( n * sizeof(T), m, (byte*) ptr, ldim * sizeof(T) );
        }
    };

    /** Legacy vector.
     * 
     * @tparam T Floating-point type
     * @tparam idx_t Index type
     * @tparam D Either Direction::Forward or Direction::Backward
     */
    template< typename T, class idx_t = std::size_t, typename int_t = internal::StrongOne, Direction D = Direction::Forward >
    struct legacyVector {
        idx_t n;                    ///< Size
        T* ptr;                     ///< Pointer to array in memory
        int_t inc;                  ///< Memory increment

        static constexpr Direction direction = D;
        
        inline constexpr T&
        operator[]( idx_t i ) const noexcept {
            assert( i >= 0);
            assert( i < n);
            return (direction == Direction::Forward)
                ? *(ptr + (i*inc))
                : *(ptr + ((n-1)-i)*inc);
        }
        
        inline constexpr legacyVector( idx_t n, T* ptr, int_t inc = 1 )
        : n(n), ptr(ptr), inc(inc)
        {
            tlapack_check_false( n < 0 );
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
     * @tparam idx_t Index type
     */
    template< typename T, class idx_t = std::size_t >
    struct legacyBandedMatrix {
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
        operator()( idx_t i, idx_t j ) const noexcept {
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
            tlapack_check_false( m < 0 );
            tlapack_check_false( n < 0 );
            tlapack_check_false( (kl + 1 > m && m > 0) || (ku + 1 > n && n > 0) );
        }
    };

} // namespace tlapack

#endif // TLAPACK_LEGACY_ARRAY_HH
