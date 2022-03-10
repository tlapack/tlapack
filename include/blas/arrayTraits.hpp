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

    // -----------------------------------------------------------------------------
    // lascl
    enum class MatrixType {
        General     = 'G',
        Lower       = 'L',
        Upper       = 'U',
        Hessenberg  = 'H',
        LowerBand   = 'B',
        UpperBand   = 'Q',
        Band        = 'Z',
    };

    // -----------------------------------------------------------------------------
    // Matrix access policies:

    enum class MatrixAccessPolicy {
        Full,
        UpperHessenberg,
        LowerHessenberg,
        UpperTriangle,
        LowerTriangle,
        StrictUpper,        // No access to the main diagonal
        StrictLower,        // No access to the main diagonal
    };

    struct full_t {
        constexpr operator Uplo() const { return Uplo::General; }
        constexpr operator MatrixType() const { return MatrixType::General; }
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::Full; }
    };

    struct upperHessenberg_t : public full_t {
        constexpr operator MatrixType() const { return MatrixType::Hessenberg; }
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::UpperHessenberg; }
    };
    struct lowerHessenberg_t : public full_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::LowerHessenberg; }
    };

    struct upperTriangle_t : public upperHessenberg_t {
        constexpr operator Uplo() const { return Uplo::Upper; }
        constexpr operator MatrixType() const { return MatrixType::Upper; }
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::UpperTriangle; }
    };
    struct lowerTriangle_t : public lowerHessenberg_t {
        constexpr operator Uplo() const { return Uplo::Lower; }
        constexpr operator MatrixType() const { return MatrixType::Lower; }
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::LowerTriangle; }
    };

    struct strictUpper_t : public upperTriangle_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::StrictUpper; }
    };
    struct strictLower_t : public lowerTriangle_t {
        constexpr operator MatrixAccessPolicy() const { return MatrixAccessPolicy::StrictLower; }
    };

    struct band_t : full_t {
        std::size_t lower_bandwidth, upper_bandwidth;

        constexpr band_t(std::size_t kl, std::size_t ku)
        : lower_bandwidth(kl), upper_bandwidth(ku)
        {}
        
        constexpr operator MatrixType() const { return MatrixType::Band; }
    };

    struct upperBand_t : public band_t, public upperTriangle_t {
        
        constexpr upperBand_t(std::size_t k) : band_t(0,k) {}

        constexpr operator MatrixType() const { return MatrixType::UpperBand; }
    };
    struct lowerBand_t : public band_t, public lowerTriangle_t {
        
        constexpr lowerBand_t(std::size_t k) : band_t(k,0) {}

        constexpr operator MatrixType() const { return MatrixType::LowerBand; }
    };

    // constants
    constexpr full_t full = { };
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
