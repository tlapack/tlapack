// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_ARRAY_TRAITS__
#define __TLAPACK_ARRAY_TRAITS__

#include <type_traits>
#include "base/types.hpp"

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

    /**
     * @brief Data type trait. Type of the entries of the array class.
     * 
     * The data type is defined on @c type_trait<array_t>::type.
     * 
     * @tparam array_t Array class.
     */
    template< class array_t > struct type_trait {
        using type = void;
    };
    
    /**
     * @brief Size type trait. Index type used in the array class.
     * 
     * The size type is defined on @c sizet_trait<array_t>::type.
     * 
     * @tparam array_t Array class.
     */
    template< class array_t > struct sizet_trait {
        using type = void;
    };

    /**
     * @brief Trait to determine if a given list of data allows optimization
     * using a optimized BLAS library.
     */
    template<class...>
    struct allow_optblas {
        static constexpr bool value = false; ///< True if the list of types
                                             ///< allows optimized BLAS library.
    };

    /// Alias for @c type_trait<>::type.
    template< class array_t >
    using type_t = typename type_trait< array_t >::type;

    /// Alias for @c sizet_trait<>::type.
    template< class array_t >
    using size_type = typename sizet_trait< array_t >::type;

    /// Alias for @c allow_optblas<>::value.
    template<class... Ts>
    constexpr bool allow_optblas_v = allow_optblas< Ts... >::value;
}

#endif // __TLAPACK_ARRAY_TRAITS__
