// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_ARRAY_TRAITS__
#define __TBLAS_ARRAY_TRAITS__

#include <type_traits>

namespace blas {
    enum class Uplo   { Upper    = 'U', Lower    = 'L', General   = 'G' };
}

namespace lapack {

    using blas::Uplo;

    /**
     * @brief Matrix access policies
     * 
     * They are used:
     * 
     * (1) on the data structure for a matrix, to state the
     * what pairs (i,j) return a valid position A(i,j).
     * 
     * (2) on the algorithm, to determine the kind of access
     * required by the algorithm.
     */
    enum class MatrixAccessPolicy {
        Dense = 'G',
        UpperHessenberg = 'H',
        LowerHessenberg = 'h',
        UpperTriangle = 'U',
        LowerTriangle = 'L',
        StrictUpper = 'S',      // No access to the main diagonal
        StrictLower = 's',      // No access to the main diagonal
    };

    /**
     * @brief Dense access
     * 
     * x x x x x
     * x x x x x
     * x x x x x
     * x x x x x
     */
    struct dense_t {
        constexpr operator Uplo() const { return Uplo::General; }
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::Dense; }
    };

    /**
     * @brief Upper Hessenberg access
     * 
     * x x x x x
     * x x x x x
     * 0 x x x x
     * 0 0 x x x
     */
    struct upperHessenberg_t : public dense_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::UpperHessenberg; }
    };

    /**
     * @brief Lower Hessenberg access
     * 
     * x x 0 0 0
     * x x x 0 0
     * x x x x 0
     * x x x x x
     */
    struct lowerHessenberg_t : public dense_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::LowerHessenberg; }
    };

    /**
     * @brief Upper Triangle access
     * 
     * x x x x x
     * 0 x x x x
     * 0 0 x x x
     * 0 0 0 x x
     */
    struct upperTriangle_t : public upperHessenberg_t {
        constexpr operator Uplo() const { return Uplo::Upper; }
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::UpperTriangle; }
    };

    /**
     * @brief Lower Triangle access
     * 
     * x 0 0 0 0
     * x x 0 0 0
     * x x x 0 0
     * x x x x 0
     */
    struct lowerTriangle_t : public lowerHessenberg_t {
        constexpr operator Uplo() const { return Uplo::Lower; }
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::LowerTriangle; }
    };

    /**
     * @brief Strict Upper Triangle access
     * 
     * 0 x x x x
     * 0 0 x x x
     * 0 0 0 x x
     * 0 0 0 0 x
     */
    struct strictUpper_t : public upperTriangle_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::StrictUpper; }
    };

    /**
     * @brief Strict Lower Triangle access
     * 
     * 0 0 0 0 0
     * x 0 0 0 0
     * x x 0 0 0
     * x x x 0 0
     */
    struct strictLower_t : public lowerTriangle_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::StrictLower; }
    };

    /**
     * @brief Band access
     * 
     * x x x 0 0
     * x x x x 0
     * 0 x x x x
     * 0 0 x x x
     */
    struct band_t : dense_t {
        std::size_t lower_bandwidth, upper_bandwidth;

        constexpr band_t(std::size_t kl, std::size_t ku)
        : lower_bandwidth(kl), upper_bandwidth(ku)
        {}
    };

    /**
     * @brief Upper Band access
     * 
     * x x x 0 0
     * 0 x x x 0
     * 0 0 x x x
     * 0 0 0 x x
     */
    struct upperBand_t : public band_t, public upperTriangle_t {
        constexpr upperBand_t(std::size_t k) : band_t(0,k) {}
    };

    /**
     * @brief Lower Band access
     * 
     * x 0 0 0 0
     * x x 0 0 0
     * 0 x x 0 0
     * 0 0 x x 0
     */
    struct lowerBand_t : public band_t, public lowerTriangle_t {
        constexpr lowerBand_t(std::size_t k) : band_t(k,0) {}
    };

    // constant expressions
    constexpr dense_t dense = { };
    constexpr upperHessenberg_t upperHessenberg = { };
    constexpr lowerHessenberg_t lowerHessenberg = { };
    constexpr upperTriangle_t upperTriangle = { };
    constexpr lowerTriangle_t lowerTriangle = { };
    constexpr strictUpper_t strictUpper = { };
    constexpr strictLower_t strictLower = { };
}

namespace blas {

    /// Data type
    template< class T > struct type_trait {};
    template< class T >
    using type_t = typename type_trait< T >::type;
    
    /// Size type
    template< class T > struct sizet_trait {};
    template< class T >
    using size_type = typename sizet_trait< T >::type;
    
    /// Layout type
    template< class T > struct layout_trait {
        using type = void;
    };
    template< class T >
    using layout_type = typename layout_trait< T >::type;

    /// Verifies if the set of data structures allow optimization using optBLAS.
    template<class...>
    struct allow_optblas {
        using type = bool; ///< Floating datatype.
        static constexpr bool value = false; ///< True if it allows optBLAS.
    };

    /// Alias for allow_optblas<>::type.
    template<class... Ts>
    using allow_optblas_t = typename allow_optblas< Ts... >::type;

    /// Alias for allow_optblas<>::value.
    template<class... Ts>
    constexpr bool allow_optblas_v = allow_optblas< Ts... >::value;
}

namespace lapack {

    using blas::type_t;
    using blas::size_type;
    using blas::layout_type;

}

#endif // __TBLAS_ARRAY_TRAITS__
