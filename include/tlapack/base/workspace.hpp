// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_WORKSPACE_HH
#define TLAPACK_WORKSPACE_HH

#include "tlapack/base/legacyArray.hpp"

namespace tlapack {

    /**
     * @brief Workspace
     * 
     * The objects of this class always maintain one of the states:
     *      1. `(n > 1) && (m > 0)`.
     *      2. `((n <= 1) || (m == 0)) && (ldim == m)`.
     */
    struct Workspace
    {
        using idx_t = std::size_t;

        // Constructors:
        
        inline constexpr
        Workspace( byte* ptr = nullptr, idx_t n = 0 ) noexcept
        : m(n), n(1), ptr(ptr), ldim(n) { }

        inline constexpr
        Workspace( const legacyMatrix<byte>& w ) noexcept
        : m(w.m), n(w.n), ptr(w.ptr), ldim(w.ldim)
        {
            if( n <= 1 || m == 0 ) ldim = m;
        }

        // Getters:
        inline constexpr byte* data() const { return ptr; }
        inline constexpr idx_t getM() const { return m; }
        inline constexpr idx_t getN() const { return n; }
        inline constexpr idx_t getLdim() const { return ldim; }
        inline constexpr idx_t size() const { return m*n; }

        /** Checks if a workspace is contiguous.
         * 
         * @note This is one of the reasons to keep the attributes protected.
         *  The objects of this class always maintain one of the states:
         *      1. (n > 1) && (m > 0).
         *      2. ((n <= 1) || (m == 0)) && (ldim == m).
         */
        inline constexpr bool isContiguous() const{ return (ldim == m); }

        /// Checks if a workspace contains the workspace of size m-by-n
        inline constexpr bool
        contains( idx_t m, idx_t n ) const 
        {    
            if( isContiguous() )
                return ( size() >= (m*n) );
            else
                return ( this->m >= m && this->n >= n );
        }

        /** Returns a workspace that is obtained by removing m*n bytes from the current one.
         * 
         * @note If the starting workspace is not contiguous, then we require:
         * 
         *          this->m == m || this->n == n
         * 
         * This is required to prevent bad partitioning by the algorithms using
         * this functionality.
         * 
         * @param m Number of rows to be extracted.
         * @param n Number of columns to be extracted.
         */
        inline constexpr Workspace
        extract( idx_t m, idx_t n ) const
        {    
            if( isContiguous() )
            {
                // contiguous space in memory
                tlapack_check( size() >= (m*n) );
                return legacyMatrix<byte>( size()-(m*n), 1, ptr + (m*n) );
            }
            else
            {
                // Only allows for the extraction of full set of rows or full set of columns 
                tlapack_check(  (this->m >= m && this->n >= n) &&
                                (this->m == m || this->n == n) );

                if( this->m == m )
                {
                    return ( this->n <= n+1 )
                        ? legacyMatrix<byte>( this->m, this->n-n, ptr + n*ldim ) // contiguous space in memory
                        : legacyMatrix<byte>( this->m, this->n-n, ptr + n*ldim, ldim ); // non-contiguous space in memory
                }
                else // if ( this->n == n )
                {
                    // non-contiguous space in memory
                    return legacyMatrix<byte>( this->m-m, this->n, ptr + m, ldim );
                }
            }
        }

    private:
        idx_t m, n; ///< Sizes
        byte* ptr;  ///< Pointer to array in memory
        idx_t ldim; ///< Leading dimension
    };
}

#endif // TLAPACK_WORKSPACE_HH