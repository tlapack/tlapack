// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_TYPES_HH__
#define __TLAPACK_TYPES_HH__

#include <complex>
#include <type_traits>

namespace tlapack {

    // -----------------------------------------------------------------------------
    // Layouts

    enum class Layout :char {
        Unspecified = 0,
        ColMajor = 'C',
        RowMajor = 'R',
        BandStorage = 'B'
    };

    // -----------------------------------------------------------------------------
    // Upper or Lower access

    enum class Uplo : char {
        Upper    = 'U', 
        Lower    = 'L', 
        General  = 'G'
    };

    // -----------------------------------------------------------------------------
    // Information about the main diagonal

    enum class Diag : char {
        NonUnit  = 'N',
        Unit     = 'U'
    };

    struct nonUnit_diagonal_t {
        constexpr operator Diag() const { return Diag::NonUnit; }
    };
    struct unit_diagonal_t {
        constexpr operator Diag() const { return Diag::Unit; }
    };

    // constants
    constexpr nonUnit_diagonal_t nonUnit_diagonal = { };
    constexpr unit_diagonal_t unit_diagonal = { };

    // -----------------------------------------------------------------------------
    // Operations over data

    enum class Op : char {
        NoTrans  = 'N',
        Trans    = 'T',
        ConjTrans = 'C',
        Conj = 0            ///< non-transpose conjugate
    };

    struct noTranspose_t {
        constexpr operator Op() const { return Op::NoTrans; }
    };
    struct transpose_t {
        constexpr operator Op() const { return Op::Trans; }
    };
    struct conjTranspose_t {
        constexpr operator Op() const { return Op::ConjTrans; }
    };

    // Constants
    constexpr noTranspose_t noTranspose = { };
    constexpr transpose_t transpose = { };
    constexpr conjTranspose_t conjTranspose = { };

    // -----------------------------------------------------------------------------
    // Sides

    enum class Side : char {
        Left     = 'L', 
        Right    = 'R'
    };

    struct left_side_t {
        constexpr operator Side() const { return Side::Left; }
    };
    struct right_side_t {
        constexpr operator Side() const { return Side::Right; }
    };

    // Constants
    constexpr left_side_t left_side { };
    constexpr right_side_t right_side { };

    // -----------------------------------------------------------------------------
    // Norm types

    enum class Norm : char {
        One = '1',  // or 'O'
        Two = '2',
        Inf = 'I',
        Fro = 'F',  // or 'E'
        Max = 'M',
    };

    struct max_norm_t {
        constexpr operator Norm() const { return Norm::Max; }
    };
    struct one_norm_t {
        constexpr operator Norm() const { return Norm::One; }
    };
    struct inf_norm_t {
        constexpr operator Norm() const { return Norm::Inf; }
    };
    struct frob_norm_t {
        constexpr operator Norm() const { return Norm::Fro; }
    };

    // Constants
    constexpr max_norm_t max_norm = { };
    constexpr one_norm_t one_norm = { };
    constexpr inf_norm_t inf_norm = { };
    constexpr frob_norm_t frob_norm = { };

    // -----------------------------------------------------------------------------
    // Directions

    enum class Direction : char {
        Forward     = 'F',
        Backward    = 'B',
    };

    struct forward_t {
        constexpr operator Direction() const { return Direction::Forward; }
    };
    struct backward_t {
        constexpr operator Direction() const { return Direction::Backward; }
    };

    // Constants
    constexpr forward_t forward { };
    constexpr backward_t backward { };

    // -----------------------------------------------------------------------------
    // Storage types

    enum class StoreV : char {
        Columnwise  = 'C',
        Rowwise     = 'R',
    };

    struct columnwise_storage_t {
        constexpr operator StoreV() const { return StoreV::Columnwise; }
    };
    struct rowwise_storage_t {
        constexpr operator StoreV() const { return StoreV::Rowwise; }
    };

    // Constants
    constexpr columnwise_storage_t columnwise_storage { };
    constexpr rowwise_storage_t rowwise_storage { };

    // -----------------------------------------------------------------------------
    // Based on C++14 std::common_type implementation from
    // http://www.cplusplus.com/reference/type_traits/std::common_type/
    // Adds promotion of complex types based on the common type of the associated
    // real types. This fixes various cases:
    //
    // std::std::common_type_t< double, complex<float> > is complex<float>  (wrong)
    //        scalar_type< double, complex<float> > is complex<double> (right)
    //
    // std::std::common_type_t< int, complex<long> > is not defined (compile error)
    //        scalar_type< int, complex<long> > is complex<long> (right)

    // for zero types
    template< typename... Types >
    struct scalar_type_traits;

    /// define scalar_type<> type alias
    template< typename... Types >
    using scalar_type = typename scalar_type_traits< Types... >::type;

    // for one type
    template< typename T >
    struct scalar_type_traits< T >
    {
        using type = typename std::decay<T>::type;
    };

    // for two types
    // relies on type of ?: operator being the common type of its two arguments
    template< typename T1, typename T2 >
    struct scalar_type_traits< T1, T2 >
    {
        using type = typename std::decay< decltype( true ? std::declval<T1>() : std::declval<T2>() ) >::type;
    };

    // for either or both complex,
    // find common type of associated real types, then add complex
    template< typename T1, typename T2 >
    struct scalar_type_traits< std::complex<T1>, T2 >
    {
        using type = std::complex< typename std::common_type< T1, T2 >::type >;
    };

    template< typename T1, typename T2 >
    struct scalar_type_traits< T1, std::complex<T2> >
    {
        using type = std::complex< typename std::common_type< T1, T2 >::type >;
    };

    template< typename T1, typename T2 >
    struct scalar_type_traits< std::complex<T1>, std::complex<T2> >
    {
        using type = std::complex< typename std::common_type< T1, T2 >::type >;
    };

    // for three or more types
    template< typename T1, typename T2, typename... Types >
    struct scalar_type_traits< T1, T2, Types... >
    {
        using type = scalar_type< scalar_type< T1, T2 >, Types... >;
    };

    // -----------------------------------------------------------------------------
    // for any combination of types, determine associated real, scalar,
    // and complex types.
    //
    // real_type< float >                               is float
    // real_type< float, double, complex<float> >       is double
    //
    // scalar_type< float >                             is float
    // scalar_type< float, complex<float> >             is complex<float>
    // scalar_type< float, double, complex<float> >     is complex<double>
    //
    // complex_type< float >                            is complex<float>
    // complex_type< float, double >                    is complex<double>
    // complex_type< float, double, complex<float> >    is complex<double>

    // for zero types
    template< typename... Types >
    struct real_type_traits;

    /// define real_type<> type alias
    template< typename... Types >
    using real_type = typename real_type_traits< Types... >::real_t;

    /// define complex_type<> type alias
    template< typename... Types >
    using complex_type = std::complex< real_type< Types... > >;

    // for one type
    template< typename T >
    struct real_type_traits<T>
    {
        using real_t = T;
    };

    // for one complex type, strip complex
    template< typename T >
    struct real_type_traits< std::complex<T> >
    {
        using real_t = T;
    };

    // for two or more types
    template< typename T1, typename... Types >
    struct real_type_traits< T1, Types... >
    {
        using real_t = scalar_type< real_type<T1>, real_type< Types... > >;
    };

} // namespace tlapack

#endif // __TLAPACK_TYPES_HH__
