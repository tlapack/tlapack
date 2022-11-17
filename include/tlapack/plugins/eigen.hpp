
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
    // Helpers

    namespace internal
    {
        /// @see https://stackoverflow.com/a/51472601/5253097
        template<class Derived>
        constexpr bool is_eigen_type_f(const Eigen::EigenBase<Derived> *) {
            return true;
        }
        constexpr bool is_eigen_type_f(const void *) {
            return false;
        }
        template<class T>
        constexpr bool is_eigen_type = is_eigen_type_f(reinterpret_cast<T*>(NULL));

        template<class Derived>
        constexpr bool is_eigen_dense_f(const Eigen::DenseBase<Derived> *) {
            return true;
        }
        constexpr bool is_eigen_dense_f(const void *) {
            return false;
        }
        template<class T>
        constexpr bool is_eigen_dense = is_eigen_dense_f(reinterpret_cast<T*>(NULL));

        template<class Derived>
        constexpr bool is_eigen_matrix_f(const Eigen::MatrixBase<Derived> *) {
            return true;
        }
        constexpr bool is_eigen_matrix_f(const void *) {
            return false;
        }
        template<class T>
        constexpr bool is_eigen_matrix = is_eigen_matrix_f(reinterpret_cast<T*>(NULL));

        template<class T>
        struct is_eigen_block_s : public std::false_type
        {};
        template<class XprType, int BlockRows, int BlockCols, bool InnerPanel>
        struct is_eigen_block_s< Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel> > : public std::true_type
        {};
        template<class T>
        constexpr bool is_eigen_block = is_eigen_block_s<T>::value;

        // template<class PlainObjectType, int MapOptions, class StrideType>
        // constexpr bool is_eigen_map_f(const Eigen::Map<PlainObjectType,MapOptions,StrideType> *) {
        //     return true;
        // }
        // constexpr bool is_eigen_map_f(const void *) {
        //     return false;
        // }
        // template<class T>
        // constexpr bool is_eigen_map = is_eigen_map_f(reinterpret_cast<T*>(NULL));
    }

    // -----------------------------------------------------------------------------
    // Data traits

    // Layout
    template< class matrix_t >
    struct LayoutImpl< matrix_t, typename std::enable_if<
        internal::is_eigen_dense<matrix_t> &&
        !internal::is_eigen_block<matrix_t>
    ,int>::type >
    {
        static constexpr Layout layout = (matrix_t::IsRowMajor)
            ? Layout::RowMajor
            : Layout::ColMajor;
    };

    // Layout
    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct LayoutImpl< Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>, int >
    {
        static constexpr Layout layout = layout<XprType>;
    };

    // Transpose type
    template< class matrix_t >
    struct transpose_type_trait< matrix_t, typename std::enable_if<
        internal::is_eigen_matrix<matrix_t>
    ,int>::type >
    {
        using type = Eigen::Matrix<
            typename matrix_t::Scalar,
            matrix_t::ColsAtCompileTime,
            matrix_t::RowsAtCompileTime, 
            ((matrix_t::IsRowMajor) ? Eigen::ColMajor : Eigen::RowMajor),
            matrix_t::MaxColsAtCompileTime,
            matrix_t::MaxRowsAtCompileTime
        >;
    };

    // // Layout
    // template<class PlainObjectType, int MapOptions, class StrideType>
    // struct LayoutImpl< Eigen::Map<PlainObjectType,MapOptions,StrideType>, int >
    // {
    //     static constexpr Layout layout = (
    //         (StrideType::InnerStrideAtCompileTime == 0)
    //             ? LayoutImpl< PlainObjectType >::layout
    //             : Layout::Unspecified
    //     );
    // };

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

    // Slice a const ref
    template< class T, class SliceSpecRow, class SliceSpecCol,
              typename std::enable_if< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int >::type = 0 >
    inline constexpr auto
    slice( const Eigen::DenseBase<T>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return A.block( rows.first, cols.first, rows.second-rows.first, cols.second-cols.first );
    }
    template<class T, typename SliceSpecCol>
    inline constexpr auto
    slice( const Eigen::DenseBase<T>& A, Eigen::Index rowIdx, SliceSpecCol&& cols ) noexcept
    {
        return A.row( rowIdx ).segment( cols.first, cols.second-cols.first );
    }
    template<class T, typename SliceSpecRow>
    inline constexpr auto
    slice( const Eigen::DenseBase<T>& A, SliceSpecRow&& rows, Eigen::Index colIdx ) noexcept
    {
        return A.col( colIdx ).segment( rows.first, rows.second-rows.first );
    }
    template<class T, typename SliceSpec>
    inline constexpr auto
    slice( const Eigen::DenseBase<T>& x, SliceSpec&& range ) noexcept
    {
        return x.segment( range.first, range.second-range.first );
    }
    template<class T, typename SliceSpec>
    inline constexpr auto
    rows( const Eigen::DenseBase<T>& A, SliceSpec&& rows ) noexcept
    {
        return A.middleRows( rows.first, rows.second-rows.first );
    }
    template<class T>
    inline constexpr auto row( const Eigen::DenseBase<T>& A, Eigen::Index rowIdx ) noexcept
    {
        return A.row( rowIdx );
    }
    template<class T, typename SliceSpec>
    inline constexpr auto cols( const Eigen::DenseBase<T>& A, SliceSpec&& cols ) noexcept
    {
        return A.middleCols( cols.first, cols.second-cols.first );
    }
    template<class T>
    inline constexpr auto col( const Eigen::DenseBase<T>& A, Eigen::Index colIdx ) noexcept
    {
        return A.col( colIdx );
    }

    // Slice a non-const ref
    template< class T, class SliceSpecRow, class SliceSpecCol,
              typename std::enable_if< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int >::type = 0 >
    inline constexpr auto
    slice( Eigen::DenseBase<T>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return A.block( rows.first, cols.first, rows.second-rows.first, cols.second-cols.first );
    }
    template<class T, typename SliceSpecCol>
    inline constexpr auto
    slice( Eigen::DenseBase<T>& A, Eigen::Index rowIdx, SliceSpecCol&& cols ) noexcept
    {
        return A.row( rowIdx ).segment( cols.first, cols.second-cols.first );
    }
    template<class T, typename SliceSpecRow>
    inline constexpr auto
    slice( Eigen::DenseBase<T>& A, SliceSpecRow&& rows, Eigen::Index colIdx ) noexcept
    {
        return A.col( colIdx ).segment( rows.first, rows.second-rows.first );
    }
    template<class T, typename SliceSpec>
    inline constexpr auto
    slice( Eigen::DenseBase<T>& x, SliceSpec&& range ) noexcept
    {
        return x.segment( range.first, range.second-range.first );
    }
    template<class T, typename SliceSpec>
    inline constexpr auto
    rows( Eigen::DenseBase<T>& A, SliceSpec&& rows ) noexcept
    {
        return A.middleRows( rows.first, rows.second-rows.first );
    }
    template<class T>
    inline constexpr auto row( Eigen::DenseBase<T>& A, Eigen::Index rowIdx ) noexcept
    {
        return A.row( rowIdx );
    }
    template<class T, typename SliceSpec>
    inline constexpr auto cols( Eigen::DenseBase<T>& A, SliceSpec&& cols ) noexcept
    {
        return A.middleCols( cols.first, cols.second-cols.first );
    }
    template<class T>
    inline constexpr auto col( Eigen::DenseBase<T>& A, Eigen::Index colIdx ) noexcept
    {
        return A.col( colIdx );
    }

    // Slice a Block
    template< class XprType, int BlockRows, int BlockCols, bool InnerPanel, class SliceSpecRow, class SliceSpecCol,
              typename std::enable_if< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int >::type = 0 >
    inline constexpr auto
    slice( Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        assert( rows.second <= A.rows() );
        assert( cols.second <= A.cols() );

        return Eigen::Block< XprType, Eigen::Dynamic, Eigen::Dynamic, InnerPanel >
        (   A.nestedExpression(),
            A.startRow() + rows.first, A.startCol() + cols.first,
            rows.second-rows.first, cols.second-cols.first );
    }
    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpecCol>
    inline constexpr auto
    slice( Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, Eigen::Index rowIdx, SliceSpecCol&& cols ) noexcept
    {
        assert( rowIdx < A.rows() );
        assert( cols.second <= A.cols() );

        return Eigen::Block< XprType, 1, Eigen::Dynamic, InnerPanel >
        (   A.nestedExpression(),
            A.startRow() + rowIdx, A.startCol() + cols.first,
            1, cols.second-cols.first );
    }
    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpecRow>
    inline constexpr auto
    slice( Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, SliceSpecRow&& rows, Eigen::Index colIdx ) noexcept
    {
        assert( rows.second <= A.rows() );
        assert( colIdx < A.cols() );

        return Eigen::Block< XprType, Eigen::Dynamic, 1, InnerPanel >
        (   A.nestedExpression(),
            A.startRow() + rows.first, A.startCol() + colIdx,
            rows.second-rows.first, 1 );
    }
    template<class VectorType, int Size, typename SliceSpec>
    inline constexpr auto
    slice( Eigen::VectorBlock<VectorType,Size>& x, SliceSpec&& range ) noexcept
    {
        return x.segment( range.first, range.second-range.first );
    }
    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpec>
    inline constexpr auto
    rows( Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, SliceSpec&& rows ) noexcept
    {
        assert( rows.second <= A.rows() );

        return Eigen::Block< XprType, Eigen::Dynamic, BlockCols, InnerPanel >
        (   A.nestedExpression(),
            A.startRow() + rows.first, A.startCol(),
            rows.second-rows.first, A.cols() );
    }
    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel>
    inline constexpr auto
    row( Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, Eigen::Index rowIdx ) noexcept
    {
        assert( rowIdx < A.rows() );

        return Eigen::Block< XprType, 1, BlockCols, InnerPanel >
        (   A.nestedExpression(),
            A.startRow() + rowIdx, A.startCol(),
            1, A.cols() );
    }
    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpec>
    inline constexpr auto
    cols( Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, SliceSpec&& cols ) noexcept
    {
        assert( cols.second <= A.cols() );

        return Eigen::Block< XprType, BlockRows, Eigen::Dynamic, InnerPanel >
        (   A.nestedExpression(),
            A.startRow(), A.startCol() + cols.first,
            A.rows(), cols.second-cols.first );
    }
    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel>
    inline constexpr auto
    col( Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, Eigen::Index colIdx ) noexcept
    {
        assert( colIdx < A.cols() );

        return Eigen::Block< XprType, BlockRows, 1, InnerPanel >
        (   A.nestedExpression(),
            A.startRow(), A.startCol() + colIdx,
            A.rows(), 1 );
    }

    #undef isSlice

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

    // -----------------------------------------------------------------------------
    // Create objects

    template<typename U>
    struct CreateImpl< U, typename std::enable_if< internal::is_eigen_dense<U> ,int>::type >
    {
        using T = typename U::Scalar;
        static constexpr int Rows_ = (U::RowsAtCompileTime == 1) ? 1 : Eigen::Dynamic;
        static constexpr int Cols_ = (U::ColsAtCompileTime == 1) ? 1 : Eigen::Dynamic;
        static constexpr int Options_ = (U::IsRowMajor)
                                        ? Eigen::RowMajor
                                        : Eigen::ColMajor;

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
        #define TLAPACK_PREFERRED_MATRIX

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

    // template< class DerivedA, class DerivedB >
    // struct matrix_type_traits< Eigen::EigenBase<DerivedA>, Eigen::EigenBase<DerivedB> >
    // {
    //     using traitsA = Eigen::internal::traits<DerivedA>;
    //     using traitsB = Eigen::internal::traits<DerivedB>;

    //     using TA = typename traitsA::Scalar;
    //     using TB = typename traitsB::Scalar;
    //     using T = scalar_type<TA,TB>;

    //     using type = Eigen::Matrix<
    //         T,
    //         (traitsA::RowsAtCompileTime == 1 && traitsB::RowsAtCompileTime == 1) ? 1 : Eigen::Dynamic,
    //         (traitsA::ColsAtCompileTime == 1 && traitsB::ColsAtCompileTime == 1) ? 1 : Eigen::Dynamic,
    //         traitsA::Options 
    //     >;
    // };

    // template<
    //     class TA, int RowsA_, int ColsA_, int OptionsA_, int MaxRowsA_, int MaxColsA_,
    //     class TB, int RowsB_, int ColsB_, int OptionsB_, int MaxRowsB_, int MaxColsB_ >
    // struct matrix_type_traits<
    //     Eigen::Matrix< TA, RowsA_, ColsA_, OptionsA_, MaxRowsA_, MaxColsA_ >,
    //     Eigen::Matrix< TB, RowsB_, ColsB_, OptionsB_, MaxRowsB_, MaxColsB_ > >
    // : public matrix_type_traits<
    //     Eigen::EigenBase< Eigen::Matrix< TA, RowsA_, ColsA_, OptionsA_, MaxRowsA_, MaxColsA_ > >,
    //     Eigen::EigenBase< Eigen::Matrix< TB, RowsB_, ColsB_, OptionsB_, MaxRowsB_, MaxColsB_ > >
    // > { };

    // template<
    //     class XprTypeA, int BlockRowsA, int BlockColsA, bool InnerPanelA,
    //     class VectorTypeB, int SizeB >
    // struct matrix_type_traits<
    //     Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA>,
    //     Eigen::VectorBlock<VectorTypeB, SizeB> >
    // : public matrix_type_traits<
    //     Eigen::EigenBase< Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >,
    //     Eigen::EigenBase< Eigen::VectorBlock<VectorTypeB, SizeB> >
    // > { };

    // template<
    //     class XprTypeA, int BlockRowsA, int BlockColsA, bool InnerPanelA,
    //     class VectorTypeB, int SizeB >
    // struct matrix_type_traits<
    //     Eigen::VectorBlock<VectorTypeB, SizeB>,
    //     Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >
    // : public matrix_type_traits<
    //     Eigen::EigenBase< Eigen::VectorBlock<VectorTypeB, SizeB> >,
    //     Eigen::EigenBase< Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >
    // > { };

    // template< class DerivedA, class DerivedB >
    // struct vector_type_traits< Eigen::EigenBase<DerivedA>, Eigen::EigenBase<DerivedB> >
    // {
    //     using traitsA = Eigen::internal::traits<DerivedA>;
    //     using traitsB = Eigen::internal::traits<DerivedB>;

    //     using TA = typename traitsA::Scalar;
    //     using TB = typename traitsB::Scalar;
    //     using T = scalar_type<TA,TB>;

    //     using type = Eigen::Matrix<
    //         T,
    //         (traitsA::RowsAtCompileTime == 1 && traitsB::RowsAtCompileTime == 1) ? 1 : Eigen::Dynamic,
    //         (traitsA::RowsAtCompileTime == 1 && traitsB::RowsAtCompileTime == 1) ? Eigen::Dynamic : 1,
    //         traitsA::Options 
    //     >;
    // };

    // template<
    //     class TA, int RowsA_, int ColsA_, int OptionsA_, int MaxRowsA_, int MaxColsA_,
    //     class TB, int RowsB_, int ColsB_, int OptionsB_, int MaxRowsB_, int MaxColsB_ >
    // struct vector_type_traits<
    //     Eigen::Matrix< TA, RowsA_, ColsA_, OptionsA_, MaxRowsA_, MaxColsA_ >,
    //     Eigen::Matrix< TB, RowsB_, ColsB_, OptionsB_, MaxRowsB_, MaxColsB_ > >
    // : public vector_type_traits<
    //     Eigen::EigenBase< Eigen::Matrix< TA, RowsA_, ColsA_, OptionsA_, MaxRowsA_, MaxColsA_ > >,
    //     Eigen::EigenBase< Eigen::Matrix< TB, RowsB_, ColsB_, OptionsB_, MaxRowsB_, MaxColsB_ > >
    // > { };

    // template<
    //     class XprTypeA, int BlockRowsA, int BlockColsA, bool InnerPanelA,
    //     class VectorTypeB, int SizeB >
    // struct vector_type_traits<
    //     Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA>,
    //     Eigen::VectorBlock<VectorTypeB, SizeB> >
    // : public vector_type_traits<
    //     Eigen::EigenBase< Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >,
    //     Eigen::EigenBase< Eigen::VectorBlock<VectorTypeB, SizeB> >
    // > { };

    // template<
    //     class XprTypeA, int BlockRowsA, int BlockColsA, bool InnerPanelA,
    //     class VectorTypeB, int SizeB >
    // struct vector_type_traits<
    //     Eigen::VectorBlock<VectorTypeB, SizeB>,
    //     Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >
    // : public vector_type_traits<
    //     Eigen::EigenBase< Eigen::VectorBlock<VectorTypeB, SizeB> >,
    //     Eigen::EigenBase< Eigen::Block<XprTypeA, BlockRowsA, BlockColsA, InnerPanelA> >
    // > { };

    /// TODO: Complete the implementation of vector_type_traits and matrix_type_traits

    // -----------------------------------------------------------------------------
    // Cast to Legacy arrays

    // template< class Derived >
    // inline constexpr auto
    // legacy_matrix( const Eigen::DenseBase<Derived>& A ) noexcept
    // {
    //     using matrix_t = Eigen::DenseBase<Derived>;
    //     using T = typename matrix_t::Scalar;
    //     using idx_t = Eigen::Index;
    //     constexpr Layout L = layout<matrix_t>;

    //     const idx_t ldim = ((Eigen::MatrixX<T>)A).outerStride();

    //     return legacyMatrix<T,idx_t,L>( nrows(A), ncols(A), (T*) ((Eigen::MatrixX<T>)A).data(), ((Eigen::MatrixX<T>)A).outerStride() );
    // }

    template< class T, int Rows, int Cols, int Options, int MaxRows, int MaxCols >
    inline constexpr auto
    legacy_matrix( const Eigen::Matrix< T, Rows, Cols, Options, MaxRows, MaxCols >& A ) noexcept
    {
        using matrix_t = Eigen::Matrix< T, Rows, Cols, Options, MaxRows, MaxCols >;
        using idx_t = Eigen::Index;
        constexpr Layout L = layout<matrix_t>;

        return legacyMatrix<T,idx_t,L>( nrows(A), ncols(A), (T*) A.data() );
    }

    template< class Derived >
    inline constexpr auto
    legacy_matrix( const Eigen::MapBase<Derived,Eigen::ReadOnlyAccessors>& A ) noexcept
    {
        using T = typename Derived::Scalar;
        using idx_t = Eigen::Index;
        constexpr Layout L = layout<Derived>;

        assert( A.innerStride() == 1 || A.innerSize() <= 1 || A.outerStride() == 1 || A.outerSize() <= 1 );

        return legacyMatrix<T,idx_t,L>( nrows(A), ncols(A), (T*) A.data(), A.outerStride() );
    }

    // template< class T, class idx_t, class int_t, Direction direction >
    // inline constexpr auto
    // legacy_matrix( const legacyVector<T,idx_t,int_t,direction>& v ) noexcept {
    //     return legacyMatrix<T,idx_t,Layout::ColMajor>( 1, v.n, v.ptr, v.inc );
    // }

    // template< class T, class idx_t, Direction direction >
    // inline constexpr auto
    // legacy_matrix( const legacyVector<T,idx_t,internal::StrongOne,direction>& v ) noexcept {
    //     return legacyMatrix<T,idx_t,Layout::ColMajor>( v.n, 1, v.ptr );
    // }

    template< class T, int Rows, int Cols, int Options, int MaxRows, int MaxCols >
    inline constexpr auto
    legacy_vector( const Eigen::Matrix< T, Rows, Cols, Options, MaxRows, MaxCols >& A ) noexcept
    {
        using idx_t = Eigen::Index;
        assert( A.innerSize() <= 1 || A.outerSize() <= 1 );
        
        return legacyVector<T,idx_t,idx_t>( A.size(), (T*) A.data(),
            (A.outerSize() == 1) ? A.innerStride() : A.outerStride() );
    }

    template< class T, int Rows, int MaxRows, int MaxCols >
    inline constexpr auto
    legacy_vector( const Eigen::Matrix< T, Rows, 1, Eigen::ColMajor, MaxRows, MaxCols >& A ) noexcept
    {
        using idx_t = Eigen::Index;
        return legacyVector<T,idx_t>( A.innerSize(), (T*) A.data() );
    }

    template< class T, int Cols, int MaxRows, int MaxCols >
    inline constexpr auto
    legacy_vector( const Eigen::Matrix< T, 1, Cols, Eigen::RowMajor, MaxRows, MaxCols >& A ) noexcept
    {
        using idx_t = Eigen::Index;
        return legacyVector<T,idx_t>( A.innerSize(), (T*) A.data() );
    }

} // namespace tlapack

#endif // TLAPACK_EIGEN_HH
