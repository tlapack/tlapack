
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_EIGEN_HH
#define TLAPACK_EIGEN_HH

#include <cassert>
#include <Eigen/Core>

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/workspace.hpp"

namespace tlapack{

    // -----------------------------------------------------------------------------
    // Data traits

    /// TODO: Implement transpose_type_trait

    // -----------------------------------------------------------------------------
    // blas functions to access Eigen properties

    // Size
    template<class T>
    inline constexpr auto
    size( const Eigen::EigenBase<T>& x ) {
        return x.size();
    }
    // Number of rows
    template<class T>
    inline constexpr auto
    nrows( const Eigen::EigenBase<T>& x ) {
        return x.rows();
    }
    // Number of columns
    template<class T>
    inline constexpr auto
    ncols( const Eigen::EigenBase<T>& x ) {
        return x.cols();
    }
    // Read policy
    template<class T>
    inline constexpr auto
    read_policy( const Eigen::DenseBase<T>& x ) {
        return dense;
    }
    // Write policy
    template<class T>
    inline constexpr auto
    write_policy( const Eigen::DenseBase<T>& x ) {
        return dense;
    }

    // -----------------------------------------------------------------------------
    // blas functions to access Eigen block operations

    #define isSlice(SliceSpec) !std::is_convertible< SliceSpec, Eigen::Index >::value

    // Slice
    template<class T, typename SliceSpecRow, typename SliceSpecCol,
        typename std::enable_if< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int >::type = 0
    >
    inline constexpr auto slice(
        const Eigen::DenseBase<T>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return A.block( rows.first, cols.first,
                        rows.second-rows.first, cols.second-cols.first );
    }
    template<class T, typename SliceSpecRow, typename SliceSpecCol,
        typename std::enable_if< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int >::type = 0
    >
    inline constexpr auto slice(
        Eigen::DenseBase<T>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return A.block( rows.first, cols.first,
                        rows.second-rows.first, cols.second-cols.first );
    }
    template< typename XprType, int BlockRows, int BlockCols, bool InnerPanel,
              typename SliceSpecRow, typename SliceSpecCol,
        typename std::enable_if< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int >::type = 0
    >
    inline constexpr auto slice(
        Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& A,
        SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return A.block( rows.first, cols.first,
                        rows.second-rows.first, cols.second-cols.first );
    }

    #undef isSlice

    // Slice
    template<class T, typename SliceSpecCol>
    inline constexpr auto slice(
        const Eigen::DenseBase<T>& A, Eigen::Index rowIdx, SliceSpecCol&& cols ) noexcept
    {
        return A.row( rowIdx ).segment( cols.first, cols.second-cols.first );
    }
    template<class T, typename SliceSpecCol>
    inline constexpr auto slice(
        Eigen::DenseBase<T>& A, Eigen::Index rowIdx, SliceSpecCol&& cols ) noexcept
    {
        return A.row( rowIdx ).segment( cols.first, cols.second-cols.first );
    }
    template< typename XprType, int BlockRows, int BlockCols, bool InnerPanel,
              typename SliceSpecCol >
    inline constexpr auto slice(
        Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& A,
        Eigen::Index rowIdx, SliceSpecCol&& cols ) noexcept
    {
        return A.row( rowIdx ).segment( cols.first, cols.second-cols.first );
    }

    // Slice
    template<class T, typename SliceSpecRow>
    inline constexpr auto slice(
        const Eigen::DenseBase<T>& A, SliceSpecRow&& rows, Eigen::Index colIdx = 0 ) noexcept
    {
        return A.col( colIdx ).segment( rows.first, rows.second-rows.first );
    }
    template<class T, typename SliceSpecRow>
    inline constexpr auto slice(
        Eigen::DenseBase<T>& A, SliceSpecRow&& rows, Eigen::Index colIdx = 0 ) noexcept
    {
        return A.col( colIdx ).segment( rows.first, rows.second-rows.first );
    }
    template< typename XprType, int BlockRows, int BlockCols, bool InnerPanel,
              typename SliceSpecRow >
    inline constexpr auto slice(
        Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& A,
        SliceSpecRow&& rows, Eigen::Index colIdx = 0 ) noexcept
    {
        return A.col( colIdx ).segment( rows.first, rows.second-rows.first );
    }

    // Rows
    template<class T, typename SliceSpec>
    inline constexpr auto rows(
        const Eigen::DenseBase<T>& A, SliceSpec&& rows ) noexcept
    {
        return A.middleRows( rows.first, rows.second-rows.first );
    }
    template<class T, typename SliceSpec>
    inline constexpr auto rows(
        Eigen::DenseBase<T>& A, SliceSpec&& rows ) noexcept
    {
        return A.middleRows( rows.first, rows.second-rows.first );
    }
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel,
             typename SliceSpec>
    inline constexpr auto rows(
        Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& A, SliceSpec&& rows ) noexcept
    {
        return A.middleRows( rows.first, rows.second-rows.first );
    }

    // Row
    template<class T>
    inline constexpr auto row( const Eigen::DenseBase<T>& A, Eigen::Index rowIdx ) noexcept
    {
        return A.row( rowIdx );
    }
    template<class T>
    inline constexpr auto row( Eigen::DenseBase<T>& A, Eigen::Index rowIdx ) noexcept
    {
        return A.row( rowIdx );
    }
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    inline constexpr auto row(
        Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& A, Eigen::Index rowIdx ) noexcept
    {
        return A.row( rowIdx );
    }

    // Cols
    template<class T, typename SliceSpec>
    inline constexpr auto cols(
        const Eigen::DenseBase<T>& A, SliceSpec&& cols ) noexcept
    {
        return A.middleCols( cols.first, cols.second-cols.first );
    }
    template<class T, typename SliceSpec>
    inline constexpr auto cols(
        Eigen::DenseBase<T>& A, SliceSpec&& cols ) noexcept
    {
        return A.middleCols( cols.first, cols.second-cols.first );
    }
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel,
             typename SliceSpec>
    inline constexpr auto cols(
        Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& A, SliceSpec&& cols ) noexcept
    {
        return A.middleCols( cols.first, cols.second-cols.first );
    }

    // Column
    template<class T>
    inline constexpr auto col( const Eigen::DenseBase<T>& A, Eigen::Index colIdx ) noexcept
    {
        return A.col( colIdx );
    }
    template<class T>
    inline constexpr auto col( Eigen::DenseBase<T>& A, Eigen::Index colIdx ) noexcept
    {
        return A.col( colIdx );
    }
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    inline constexpr auto col( Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& A, Eigen::Index colIdx ) noexcept
    {
        return A.col( colIdx );
    }

    // Diagonal
    template<class T, class int_t>
    inline constexpr auto diag( const Eigen::MatrixBase<T>& A, int_t diagIdx = 0 ) noexcept
    {
        return A.diagonal( diagIdx );
    }
    template<class T, class int_t>
    inline constexpr auto diag( Eigen::MatrixBase<T>& A, int_t diagIdx = 0 ) noexcept
    {
        return A.diagonal( diagIdx );
    }
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpec, class int_t>
    inline constexpr auto diag( Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& A, int_t diagIdx = 0 ) noexcept
    {
        return A.diagonal( diagIdx );
    }

    // -----------------------------------------------------------------------------
    // Create objects

    template<typename U>
    struct CreateImpl< U, typename std::enable_if<
        !std::is_same< decltype(std::declval<U>().derived()), void >::value
    ,int>::type >
    {
        using traits = Eigen::internal::traits<U>;
        using T = typename traits::Scalar;
        static constexpr int Rows_ = (traits::RowsAtCompileTime == 1) ? 1 : Eigen::Dynamic;
        static constexpr int Cols_ = (traits::ColsAtCompileTime == 1) ? 1 : Eigen::Dynamic;
        static constexpr int Options_ = traits::Options;

        using matrix_t  = Eigen::Matrix< T, Rows_, Cols_, Options_ >;
        using idx_t     = Eigen::Index;

        inline constexpr auto
        operator()( std::vector<T>& v, idx_t m, idx_t n = 1 ) const {
            assert( m >= 0 && n >= 0 );
            v.resize(0);
            return matrix_t( m, n );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n, Workspace& rW ) const {
        
            using map_t = Eigen::Map< matrix_t, Eigen::Unaligned, Eigen::OuterStride<> >;

            assert( m >= 0 && n >= 0 );

            if( matrix_t::IsRowMajor )
            {
                rW = W.extract( n*sizeof(T), m );
                return map_t( (T*) W.data(), m, n, Eigen::OuterStride<>(
                                (W.isContiguous()) ? n : W.getLdim()/sizeof(T)
                              ) );
            }
            else
            {
                rW = W.extract( m*sizeof(T), n );
                return map_t( (T*) W.data(), m, n, Eigen::OuterStride<>(
                                (W.isContiguous()) ? m : W.getLdim()/sizeof(T)
                              ) );
            }
        }
        inline constexpr auto
        operator()( const Workspace& W, idx_t m, Workspace& rW ) const {
            return operator()( W, m, 1, rW );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n = 1 ) const {
        
            using map_t = Eigen::Map< matrix_t, Eigen::Unaligned, Eigen::OuterStride<> >;
            
            assert( m >= 0 && n >= 0 );

            if( matrix_t::IsRowMajor )
            {
                tlapack_check( W.contains( n*sizeof(T), m ) );
                return map_t( (T*) W.data(), m, n, Eigen::OuterStride<>(
                                (W.isContiguous()) ? n : W.getLdim()/sizeof(T)
                              ) );
            }
            else
            {
                tlapack_check( W.contains( m*sizeof(T), n ) );
                return map_t( (T*) W.data(), m, n, Eigen::OuterStride<>(
                                (W.isContiguous()) ? m : W.getLdim()/sizeof(T)
                              ) );
            }
        }
    };

    // Matrix and vector type specialization:

    #ifndef TLAPACK_PREFERRED_MATRIX

        // for two types
        // should be especialized for every new matrix class
        template< typename matrixA_t, typename matrixB_t >
        struct matrix_type_traits< matrixA_t, matrixB_t >
        {
            using T = scalar_type< type_t<matrixA_t>, type_t<matrixB_t> >;

            static constexpr Layout LA = layout<matrixA_t>;
            static constexpr Layout LB = layout<matrixB_t>;
            static constexpr int L =
                ((LA == Layout::RowMajor) && (LB == Layout::RowMajor))
                    ? Eigen::RowMajor
                    : Eigen::ColMajor;

            using type = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, L >;
        };

        // for two types
        // should be especialized for every new vector class
        template< typename vecA_t, typename vecB_t >
        struct vector_type_traits< vecA_t, vecB_t >
        {
            using T = scalar_type< type_t<vecA_t>, type_t<vecB_t> >;
            using type = Eigen::Matrix< T, Eigen::Dynamic, 1 >;
        };

    #endif // TLAPACK_PREFERRED_MATRIX

    template< class DerivedA, class DerivedB >
    struct matrix_type_traits< Eigen::EigenBase<DerivedA>, Eigen::EigenBase<DerivedB> >
    {
        using traitsA = Eigen::internal::traits<DerivedA>;
        using traitsB = Eigen::internal::traits<DerivedB>;

        using TA = typename traitsA::Scalar;
        using TB = typename traitsB::Scalar;
        using T = scalar_type<TA,TB>;

        using type = Eigen::Matrix<
            T,
            (traitsA::RowsAtCompileTime == 1 && traitsB::RowsAtCompileTime == 1) ? 1 : Eigen::Dynamic,
            (traitsA::ColsAtCompileTime == 1 && traitsB::ColsAtCompileTime == 1) ? 1 : Eigen::Dynamic,
            traitsA::Options 
        >;
    };

    template<
        class TA, int RowsA_, int ColsA_, int OptionsA_, int MaxRowsA_, int MaxColsA_,
        class TB, int RowsB_, int ColsB_, int OptionsB_, int MaxRowsB_, int MaxColsB_ >
    struct matrix_type_traits<
        Eigen::Matrix< TA, RowsA_, ColsA_, OptionsA_, MaxRowsA_, MaxColsA_ >,
        Eigen::Matrix< TB, RowsB_, ColsB_, OptionsB_, MaxRowsB_, MaxColsB_ > >
    : public matrix_type_traits<
        Eigen::EigenBase< Eigen::Matrix< TA, RowsA_, ColsA_, OptionsA_, MaxRowsA_, MaxColsA_ > >,
        Eigen::EigenBase< Eigen::Matrix< TB, RowsB_, ColsB_, OptionsB_, MaxRowsB_, MaxColsB_ > >
    > { };

    template<
        class XprTypeA, int BlockRowsA, int BlockColsA, bool InnerPanelA,
        class VectorTypeB, int SizeB >
    struct matrix_type_traits<
        Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA>,
        Eigen::VectorBlock<VectorTypeB, SizeB> >
    : public matrix_type_traits<
        Eigen::EigenBase< Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >,
        Eigen::EigenBase< Eigen::VectorBlock<VectorTypeB, SizeB> >
    > { };

    template<
        class XprTypeA, int BlockRowsA, int BlockColsA, bool InnerPanelA,
        class VectorTypeB, int SizeB >
    struct matrix_type_traits<
        Eigen::VectorBlock<VectorTypeB, SizeB>,
        Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >
    : public matrix_type_traits<
        Eigen::EigenBase< Eigen::VectorBlock<VectorTypeB, SizeB> >,
        Eigen::EigenBase< Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >
    > { };

    template< class DerivedA, class DerivedB >
    struct vector_type_traits< Eigen::EigenBase<DerivedA>, Eigen::EigenBase<DerivedB> >
    {
        using traitsA = Eigen::internal::traits<DerivedA>;
        using traitsB = Eigen::internal::traits<DerivedB>;

        using TA = typename traitsA::Scalar;
        using TB = typename traitsB::Scalar;
        using T = scalar_type<TA,TB>;

        using type = Eigen::Matrix<
            T,
            (traitsA::RowsAtCompileTime == 1 && traitsB::RowsAtCompileTime == 1) ? 1 : Eigen::Dynamic,
            (traitsA::RowsAtCompileTime == 1 && traitsB::RowsAtCompileTime == 1) ? Eigen::Dynamic : 1,
            traitsA::Options 
        >;
    };

    template<
        class TA, int RowsA_, int ColsA_, int OptionsA_, int MaxRowsA_, int MaxColsA_,
        class TB, int RowsB_, int ColsB_, int OptionsB_, int MaxRowsB_, int MaxColsB_ >
    struct vector_type_traits<
        Eigen::Matrix< TA, RowsA_, ColsA_, OptionsA_, MaxRowsA_, MaxColsA_ >,
        Eigen::Matrix< TB, RowsB_, ColsB_, OptionsB_, MaxRowsB_, MaxColsB_ > >
    : public vector_type_traits<
        Eigen::EigenBase< Eigen::Matrix< TA, RowsA_, ColsA_, OptionsA_, MaxRowsA_, MaxColsA_ > >,
        Eigen::EigenBase< Eigen::Matrix< TB, RowsB_, ColsB_, OptionsB_, MaxRowsB_, MaxColsB_ > >
    > { };

    template<
        class XprTypeA, int BlockRowsA, int BlockColsA, bool InnerPanelA,
        class VectorTypeB, int SizeB >
    struct vector_type_traits<
        Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA>,
        Eigen::VectorBlock<VectorTypeB, SizeB> >
    : public vector_type_traits<
        Eigen::EigenBase< Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >,
        Eigen::EigenBase< Eigen::VectorBlock<VectorTypeB, SizeB> >
    > { };

    template<
        class XprTypeA, int BlockRowsA, int BlockColsA, bool InnerPanelA,
        class VectorTypeB, int SizeB >
    struct vector_type_traits<
        Eigen::VectorBlock<VectorTypeB, SizeB>,
        Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >
    : public vector_type_traits<
        Eigen::EigenBase< Eigen::VectorBlock<VectorTypeB, SizeB> >,
        Eigen::EigenBase< Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >
    > { };

    /// TODO: Complete the implementation of vector_type_traits and matrix_type_traits

} // namespace tlapack

#endif // TLAPACK_EIGEN_HH
