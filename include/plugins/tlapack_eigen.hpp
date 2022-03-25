
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_EIGEN_HH__
#define __TLAPACK_EIGEN_HH__

#include <Eigen/Core>
#include <type_traits>
#include "blas/arrayTraits.hpp"

namespace blas{

    // // -----------------------------------------------------------------------------
    // // is_EigenBlock
    
    // template< class >
    // struct is_EigenBlock : public std::false_type {};

    // template< typename T, int R, int C, bool P >
    // struct is_EigenBlock< Eigen::Block<T,R,C,P> > : public std::true_type {};

    // -----------------------------------------------------------------------------
    // Data traits for Eigen

    // Data type
    template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
    struct type_trait< Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> > {
        using type = Scalar_;
    };
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct type_trait< Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel> > {
        using type = typename XprType::Scalar;
    };
    template<class T>
    struct type_trait< Eigen::VectorBlock<T> > {
        using type = typename T::Scalar;
    };
    template<class T, int idx>
    struct type_trait< Eigen::Diagonal<T,idx> > {
        using type = typename T::Scalar;
    };

    // Size type
    template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
    struct sizet_trait< Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> > {
        using type = Eigen::Index;
    };
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct sizet_trait< Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel> > {
        using type = Eigen::Index;
    };
    template<class T>
    struct sizet_trait< Eigen::VectorBlock<T> > {
        using type = Eigen::Index;
    };
    template<class T, int idx>
    struct sizet_trait< Eigen::Diagonal<T,idx> > {
        using type = Eigen::Index;
    };

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
        return lapack::dense;
    }
    // Write policy
    template<class T>
    inline constexpr auto
    write_policy( const Eigen::DenseBase<T>& x ) {
        return lapack::dense;
    }

    // -----------------------------------------------------------------------------
    // blas functions to access Eigen block operations

    // Submatrix
    template<class T, typename SliceSpecRow, typename SliceSpecCol>
    inline constexpr auto submatrix(
        const Eigen::DenseBase<T>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return A.block( rows.first, cols.first,
                        rows.second-rows.first, cols.second-cols.first );
    }
    template<class T, typename SliceSpecRow, typename SliceSpecCol>
    inline constexpr auto submatrix(
        Eigen::DenseBase<T>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return A.block( rows.first, cols.first,
                        rows.second-rows.first, cols.second-cols.first );
    }
    template< typename XprType, int BlockRows, int BlockCols, bool InnerPanel,
              typename SliceSpecRow, typename SliceSpecCol >
    inline constexpr auto submatrix(
        Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& A,
        SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return A.block( rows.first, cols.first,
                        rows.second-rows.first, cols.second-cols.first );
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

    // Subvector
    template<class T, typename SliceSpec>
    inline constexpr auto subvector( const Eigen::DenseBase<T>& v, SliceSpec&& rows ) noexcept
    {
        return v.segment( rows.first, rows.second-rows.first );
    }
    template<class T, typename SliceSpec>
    inline constexpr auto subvector( Eigen::DenseBase<T>& v, SliceSpec&& rows ) noexcept
    {
        return v.segment( rows.first, rows.second-rows.first );
    }
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpec>
    inline constexpr auto subvector( Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& v, SliceSpec&& rows ) noexcept
    {
        return v.segment( rows.first, rows.second-rows.first );
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

} // namespace blas

namespace lapack {

    using blas::size;
    using blas::nrows;
    using blas::ncols;
    using blas::read_policy;
    using blas::write_policy;

    using blas::submatrix;
    using blas::rows;
    using blas::row;
    using blas::cols;
    using blas::col;
    using blas::subvector;
    using blas::diag;

} // namespace lapack

#endif // __TLAPACK_EIGEN_HH__