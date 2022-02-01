
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_EIGEN_HH__
#define __TLAPACK_EIGEN_HH__

#include <Eigen/Dense>
#include <type_traits>

namespace blas{

    // -----------------------------------------------------------------------------
    // enable_if_t is defined in C++14; here's a C++11 definition
    #if __cplusplus >= 201402L
        using std::enable_if_t;
    #else
        template< bool B, class T = void >
        using enable_if_t = typename enable_if<B,T>::type;
    #endif

    // -----------------------------------------------------------------------------
    // is_EigenBlock
    
    template< class >
    struct is_EigenBlock : public std::false_type {};

    template< typename T, int R, int C, bool P >
    struct is_EigenBlock< Eigen::Block<T,R,C,P> > : public std::true_type {};

    // -----------------------------------------------------------------------------
    // Data traits for Eigen

    #ifndef TBLAS_ARRAY_TRAITS
        #define TBLAS_ARRAY_TRAITS

        // Data type
        template< class T > struct type_trait {};
        template< class T >
        using type_t = typename type_trait< T >::type;

        // Size type
        template< class T > struct sizet_trait {};
        template< class T >
        using size_type = typename sizet_trait< T >::type;

    #endif // TBLAS_ARRAY_TRAITS

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
    template<class T>
    inline constexpr auto diag( const Eigen::MatrixBase<T>& A, int diagIdx = 0 ) noexcept
    {
        return A.diagonal( diagIdx );
    }
    template<class T>
    inline constexpr auto diag( Eigen::MatrixBase<T>& A, int diagIdx = 0 ) noexcept
    {
        return A.diagonal( diagIdx );
    }
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpec>
    inline constexpr auto diag( Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& A, int diagIdx = 0 ) noexcept
    {
        return A.diagonal( diagIdx );
    }

} // namespace blas

namespace lapack {
    
    using blas::type_trait;
    using blas::sizet_trait;

    using blas::size;
    using blas::nrows;
    using blas::ncols;

    using blas::submatrix;
    using blas::rows;
    using blas::row;
    using blas::cols;
    using blas::col;
    using blas::subvector;
    using blas::diag;

} // namespace lapack

#endif // __TLAPACK_EIGEN_HH__