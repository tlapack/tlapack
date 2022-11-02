// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ARRAY_TRAITS
#define TLAPACK_ARRAY_TRAITS

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
     * using T = tlapack::type_t<matrix_t>;
     * using idx_t = tlapack::size_type<matrix_t>;
     * 
     * tlapack::Create<matrix_t> new_matrix; // Creates the functor
     * 
     * idx_t m = 11;
     * idx_t n = 6;
     * 
     * std::vector<T> A_container; // Empty vector
     * auto A = new_matrix(A_container, m, n); // Initialize A_container if needed
     * 
     * tlapack::vectorOfBytes B_container; // Empty vector
     * auto B = new_matrix(
     *  tlapack::alloc_workspace( B_container, m*n*sizeof(T) ),
     *  m, n ); // B_container stores allocated memory
     * 
     * tlapack::vectorOfBytes C_container; // Empty vector
     * Workspace W; // Empty workspace
     * auto C = new_matrix(
     *  tlapack::alloc_workspace( C_container, m*n*sizeof(T) ),
     *  m, n, W ); // W receives the updated workspace, i.e., without the space taken by C
     * @endcode
     */
    template< class matrix_t, class = int > struct CreateImpl
    {
        static_assert(false && sizeof(matrix_t), "Must use correct specialization");

        /**
         * @brief Creates an object (matrix or vector)
         * 
         * @param[out] v
         *      Vector that can be used to reference allocated memory.
         * @param[in] m Number of rows
         * @param[in] n Number of Columns
         * 
         * @return The new object
         */
        template< class T, class idx_t >
        inline constexpr auto
        operator()( std::vector<T>& v, idx_t m, idx_t n ) const {
            return matrix_t();
        }

        /**
         * @brief Creates an object (matrix or vector)
         * 
         * @param[in] W
         *      Workspace that references to allocated memory.
         * @param[in] m Number of rows
         * @param[in] n Number of Columns
         * @param[out] rW
         *      On exit, receives the updated workspace, i.e., that references
         *      remaining allocated memory.
         * 
         * @return The new object
         */
        template< class idx_t, class Workspace >
        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n, Workspace& rW ) const {
            return matrix_t();
        }

        /**
         * @brief Creates an object (matrix or vector)
         * 
         * @param[in] W
         *      Vector that can be used to reference allocated memory.
         * @param[in] m Number of rows
         * @param[in] n Number of Columns
         * 
         * @return The new object
         */
        template< class idx_t, class Workspace >
        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n ) const {
            return matrix_t();
        }
    };

    template< class T > using Create = CreateImpl<T,int>;

    //--------------------------------------------------------------------------
    // Transpose type trait

    template< class T >
    struct transpose_type_trait;

    template< class T >
    using transpose_type = typename transpose_type_trait<T>::type;

    //--------------------------------------------------------------------------
    // Common matrix type deduction

    template< typename... matrix_t >
    struct matrix_type_traits;

    /// define @c matrix_type<>::type alias
    template< typename... matrix_t >
    using matrix_type = typename matrix_type_traits< matrix_t... >::type;

    // for one type
    /// TODO: Verify that matrix_t is actually a matrix type
    template< typename matrix_t >
    struct matrix_type_traits< matrix_t >
    {
        using type = typename std::decay<matrix_t>::type;
    };

    // for three or more types
    template< typename matrixA_t, typename matrixB_t, typename... matrix_t >
    struct matrix_type_traits< matrixA_t, matrixB_t, matrix_t... >
    {
        using type = matrix_type< matrix_type< matrixA_t, matrixB_t >, matrix_t... >;
    };

    //--------------------------------------------------------------------------
    // Common vector type deduction

    template< typename... vector_t >
    struct vector_type_traits;

    /// define @c vector_type<>::type alias
    template< typename... vector_t >
    using vector_type = typename vector_type_traits< vector_t... >::type;

    // for one type
    /// TODO: Verify that vector_t is actually a vector type
    template< typename vector_t >
    struct vector_type_traits< vector_t >
    {
        using type = typename std::decay<vector_t>::type;
    };

    // for three or more types
    template< typename vectorA_t, typename vectorB_t, typename... vector_t >
    struct vector_type_traits< vectorA_t, vectorB_t, vector_t... >
    {
        using type = vector_type< vector_type< vectorA_t, vectorB_t >, vector_t... >;
    };
}

#endif // TLAPACK_ARRAY_TRAITS
