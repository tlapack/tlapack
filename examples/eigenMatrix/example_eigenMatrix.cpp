/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#define NDEBUG 1

#include <Eigen/Dense>

namespace blas{
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

    // Submatrix
    template<class T, typename SliceSpecRow, typename SliceSpecCol>
    inline constexpr auto submatrix(
        Eigen::DenseBase<T>& A,
        const SliceSpecRow& rows,
        const SliceSpecCol& cols ) noexcept
    {
        return A.block( rows.first, cols.first,
                        rows.second-rows.first, cols.second-cols.first );
    }

    // Extract column from matrix
    template<class T, typename SliceSpecRow>
    inline constexpr auto col(
        Eigen::DenseBase<T>& A, size_t colIdx, const SliceSpecRow& rows ) noexcept
    {
        return A.col( colIdx ).segment( rows.first, rows.second-rows.first );
    }

    // Subvector
    template<class T, typename SliceSpec>
    inline constexpr auto subvector(
        Eigen::DenseBase<T>& A, const SliceSpec& rows ) noexcept
    {
        return A.segment( rows.first, rows.second-rows.first );
    }
}

#include <tlapack.hpp>
#include <tblas.hpp>

namespace blas{
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
}

#include <iostream>

int main( int argc, char** argv )
{
    using pair = std::pair<size_t,size_t>;
    using blas::submatrix;

    const size_t m = 5;
    const size_t n = 3;
    constexpr size_t k = (m <= n) ? m : n;

    Eigen::Matrix<float, m, n> A {
        { 1,  2,  3},
        { 4,  5,  6},
        { 7,  8,  9},
        {10, 11, 12},
        {13, 14, 15}
    };
    Eigen::Matrix<float, n, n> R = Eigen::Matrix<float, n, n>::Zero();
    Eigen::Matrix<float, k, 1> tau;
    Eigen::Matrix<float, n-1, 1> work;
    Eigen::Matrix<float, m, n> QtimesR = Eigen::Matrix<float, m, n>::Zero();

    std::cout << A << std::endl;

    lapack::geqr2( A, tau, work );
    lapack::lacpy(  lapack::upper_triangle, 
                    submatrix(A,pair(0,k),pair(0,k)), 
                    R );
    lapack::org2r( k, A, tau, work );

    std::cout << A << std::endl;
    std::cout << R << std::endl;

    blas::gemm( blas::Op::NoTrans, blas::Op::NoTrans,
        1.0, A, R, 0.0, QtimesR);

    std::cout << QtimesR << std::endl;

    return 0;
}
