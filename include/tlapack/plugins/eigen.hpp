
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_EIGEN_HH
#define TLAPACK_EIGEN_HH

#include <Eigen/Core>
#include "tlapack/base/legacyArray.hpp"

namespace tlapack{

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

    template<typename T, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
    struct Create< Eigen::Matrix< T, Rows_, Cols_, Options_, MaxRows_, MaxCols_ > >
    {
        using matrix_t  = Eigen::Matrix< T, Eigen::Dynamic, (MaxCols_==1) ? 1 : Eigen::Dynamic, Options_ >;
        using idx_t     = Eigen::Index;

        inline constexpr auto
        operator()( T* ptr, idx_t m, idx_t n ) const {
            return Eigen::Map< matrix_t >( ptr, m, n );
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n, Workspace& rW ) const {
        
            using map_t = Eigen::Map< matrix_t, Eigen::Unaligned, Eigen::OuterStride<> >;

            T* ptr = (T*) W.ptr;
            if( !matrix_t::IsRowMajor )
            {
                rW = W.extract( m*sizeof(T), n );
                return map_t( ptr, m, n, Eigen::OuterStride<>(
                                (rW.ldim == rW.m) ? m : rW.ldim/sizeof(T)
                              ) );
            }
            else
            {
                rW = W.extract( n*sizeof(T), m );
                return map_t( ptr, m, n, Eigen::OuterStride<>(
                                (rW.ldim == rW.m) ? n : rW.ldim/sizeof(T)
                              ) );
            }
        }

        inline constexpr auto
        operator()( const Workspace& W, idx_t m, idx_t n ) const {
            Workspace rW;
            return operator()( W, m, n, rW );
        }
    };
    
    // Other types that may appear

    template<typename PlainObjectType, int MapOptions, typename StrideType>
    struct Create< Eigen::Map<PlainObjectType, MapOptions, StrideType> >
    : public Create<PlainObjectType> { };

    template<typename MatrixType, int DiagIndex_>
    struct Create< Eigen::Diagonal< MatrixType, DiagIndex_ > >
    : public Create<MatrixType> { };

    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct Create< Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel> >
    : public Create<XprType> { };

} // namespace tlapack

#endif // TLAPACK_EIGEN_HH
