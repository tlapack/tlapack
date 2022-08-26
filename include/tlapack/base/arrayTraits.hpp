// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ARRAY_TRAITS
#define TLAPACK_ARRAY_TRAITS

#include <type_traits>
#include "tlapack/base/types.hpp"

namespace tlapack {
    
    /**
     * @brief Layout trait. Layout type used in the array class.
     * 
     * @tparam array_t Array class.
     */
    template< class array_t >
    constexpr Layout layout = Layout::Unspecified;

    // -----------------------------------------------------------------------------
    // Access policies

    /**
     * @brief Matrix access policies
     * 
     * They are used:
     * 
     * (1) on the data structure for a matrix, to state the what pairs (i,j)
     * return a valid position A(i,j), where A is a m-by-n matrix.
     * 
     * (2) on the algorithm, to determine the kind of access
     * required by the algorithm.
     */
    enum class MatrixAccessPolicy {
        Dense = 'G',            ///< 0 <= i <= m,   0 <= j <= n.
        UpperHessenberg = 'H',  ///< 0 <= i <= j+1, 0 <= j <= n.
        LowerHessenberg = 'h',  ///< 0 <= i <= m,   0 <= j <= i+1.
        UpperTriangle = 'U',    ///< 0 <= i <= j,   0 <= j <= n.
        LowerTriangle = 'L',    ///< 0 <= i <= m,   0 <= j <= i.
        StrictUpper = 'S',      ///< 0 <= i <= j-1, 0 <= j <= n.
        StrictLower = 's',      ///< 0 <= i <= m,   0 <= j <= i-1.
    };

    /**
     * @brief Dense access.
     * 
     * Pairs (i,j) such that 0 <= i <= m,   0 <= j <= n     in a m-by-n matrix.
     * 
     *      x x x x x
     *      x x x x x
     *      x x x x x
     *      x x x x x
     */
    struct dense_t {
        constexpr operator Uplo() const { return Uplo::General; }
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::Dense; }
    };

    /**
     * @brief Upper Hessenberg access
     * 
     * Pairs (i,j) such that 0 <= i <= j+1, 0 <= j <= n     in a m-by-n matrix.
     * 
     *      x x x x x
     *      x x x x x
     *      0 x x x x
     *      0 0 x x x
     */
    struct upperHessenberg_t : public dense_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::UpperHessenberg; }
    };

    /**
     * @brief Lower Hessenberg access
     * 
     * Pairs (i,j) such that 0 <= i <= m,   0 <= j <= i+1   in a m-by-n matrix.
     * 
     *      x x 0 0 0
     *      x x x 0 0
     *      x x x x 0
     *      x x x x x
     */
    struct lowerHessenberg_t : public dense_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::LowerHessenberg; }
    };

    /**
     * @brief Upper Triangle access
     * 
     * Pairs (i,j) such that 0 <= i <= j,   0 <= j <= n     in a m-by-n matrix.
     * 
     *      x x x x x
     *      0 x x x x
     *      0 0 x x x
     *      0 0 0 x x
     */
    struct upperTriangle_t : public upperHessenberg_t {
        constexpr operator Uplo() const { return Uplo::Upper; }
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::UpperTriangle; }
    };

    /**
     * @brief Lower Triangle access
     * 
     * Pairs (i,j) such that 0 <= i <= m,   0 <= j <= i     in a m-by-n matrix.
     * 
     *      x 0 0 0 0
     *      x x 0 0 0
     *      x x x 0 0
     *      x x x x 0
     */
    struct lowerTriangle_t : public lowerHessenberg_t {
        constexpr operator Uplo() const { return Uplo::Lower; }
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::LowerTriangle; }
    };

    /**
     * @brief Strict Upper Triangle access
     * 
     * Pairs (i,j) such that 0 <= i <= j-1, 0 <= j <= n     in a m-by-n matrix.
     * 
     *      0 x x x x
     *      0 0 x x x
     *      0 0 0 x x
     *      0 0 0 0 x
     */
    struct strictUpper_t : public upperTriangle_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::StrictUpper; }
    };

    /**
     * @brief Strict Lower Triangle access
     * 
     * Pairs (i,j) such that 0 <= i <= m,   0 <= j <= i-1   in a m-by-n matrix.
     * 
     *      0 0 0 0 0
     *      x 0 0 0 0
     *      x x 0 0 0
     *      x x x 0 0
     */
    struct strictLower_t : public lowerTriangle_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::StrictLower; }
    };

    /**
     * @brief Band access
     * 
     * Pairs (i,j) such that max(0,j-ku) <= i <= min(m,j+kl) in a m-by-n matrix,
     * where kl is the lower_bandwidth and ku is the upper_bandwidth.
     * 
     *      x x x 0 0
     *      x x x x 0
     *      0 x x x x
     *      0 0 x x x
     */
    struct band_t : dense_t {
        std::size_t lower_bandwidth; ///< Number of subdiagonals.
        std::size_t upper_bandwidth; ///< Number of superdiagonals.

        constexpr band_t(std::size_t kl, std::size_t ku)
        : lower_bandwidth(kl), upper_bandwidth(ku)
        {}
    };

    // constant expressions
    constexpr dense_t dense = { };
    constexpr upperHessenberg_t upperHessenberg = { };
    constexpr lowerHessenberg_t lowerHessenberg = { };
    constexpr upperTriangle_t upperTriangle = { };
    constexpr lowerTriangle_t lowerTriangle = { };
    constexpr strictUpper_t strictUpper = { };
    constexpr strictLower_t strictLower = { };

    // Functor:

    /**
     * @brief Implements the options for data creation in <T>LAPACK
     * 
     * Must be specialized for each datatype
     * 
     * Usage:
     * @code{.cpp}
     * // matrix_t is a predefined type at this point
     * 
     * using T = type_t<matrix_t>;
     * using idx_t = size_type<matrix_t>;
     * 
     * Create<matrix_t> new_matrix; // Creates the functor
     * 
     * idx_t m = 11;
     * idx_t n = 6;
     * 
     * std::vector<T> A_container( m*n ); // Pre-allocated space
     * auto A = new_matrix(&A_container[0], 11, 6);
     * 
     * std::vector<T> B_container; // Empty vector
     * auto B = new_matrix(11, 6, B_container); // Allocates space
     * 
     * tlapack::vectorOfBytes C_container; // Empty vector of bytes
     * auto C = new_matrix(11, 6, C_container); // Allocates space
     * @endcode
     */
    template< class matrix_t > class Create {

        static_assert(false && sizeof(matrix_t), "Must use correct specialization");

        /**
         * @brief Creates a matrix using preallocated memory
         * 
         * @param[in] ptr   Pointer to the preallocated memory
         * @param[in] m     Number of rows
         * @param[in] n     Number of Columns
         * 
         * @return The matrix using the preallocated memory
         */
        template< class T, class idx_t >
        inline constexpr auto
        operator()( T* ptr, idx_t m, idx_t n ) const {
            return matrix_t();
        }

        /**
         * @brief Creates a matrix using preallocated memory
         * 
         * @param[in] ptr   Pointer to the preallocated memory
         * @param[in] m     Number of rows
         * @param[in] n     Number of Columns
         * @param[in] size  Preallocated size, for error checks
         * 
         * @return The matrix using the preallocated memory
         */
        template< class T, class idx_t >
        inline constexpr auto
        operator()( T* ptr, idx_t m, idx_t n, size_t size ) const {
            assert( size >= m*n );
            return matrix_t();
        }

        /**
         * @brief Creates a matrix
         * 
         * @param[in] m Number of rows
         * @param[in] n Number of Columns
         * @param[out] container
         *      Vector that can be used to store some allocated memory.
         * 
         * @return The new matrix
         */
        template< class T, class idx_t, class Allocator >
        inline constexpr auto
        operator()( idx_t m, idx_t n, std::vector<T,Allocator>& container ) const {
            return matrix_t();
        }

        /**
         * @brief Creates a matrix
         * 
         * @param[in] m Number of rows
         * @param[in] n Number of Columns
         * @param[out] container
         *      Vector that can be used to store some allocated memory.
         * 
         * @return The new matrix
         */
        template< class idx_t >
        inline constexpr auto
        operator()( idx_t m, idx_t n, vectorOfBytes& container ) const {
            return matrix_t();
        }
    };
}

#endif // TLAPACK_ARRAY_TRAITS
