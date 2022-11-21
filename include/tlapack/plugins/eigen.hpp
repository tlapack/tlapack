
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
        // Auxiliary constexpr routines
        template<class Derived>
        inline constexpr bool is_eigen_type_f(const Eigen::EigenBase<Derived> *) {
            return true;
        }
        inline constexpr bool is_eigen_type_f(const void *) {
            return false;
        }
        template<class Derived>
        inline constexpr bool is_eigen_dense_f(const Eigen::DenseBase<Derived> *) {
            return true;
        }
        inline constexpr bool is_eigen_dense_f(const void *) {
            return false;
        }
        template<class Derived>
        inline constexpr bool is_eigen_matrix_f(const Eigen::MatrixBase<Derived> *) {
            return true;
        }
        inline constexpr bool is_eigen_matrix_f(const void *) {
            return false;
        }
        template<class XprType, int BlockRows, int BlockCols, bool InnerPanel>
        inline constexpr bool is_eigen_block_f(const Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel> *) {
            return true;
        }
        inline constexpr bool is_eigen_block_f(const void *) {
            return false;
        }
        template<class PlainObjectType, int MapOptions, class StrideType>
        inline constexpr bool is_eigen_map_f(const Eigen::Map<PlainObjectType,MapOptions,StrideType> *) {
            return true;
        }
        inline constexpr bool is_eigen_map_f(const void *) {
            return false;
        }

        /// True if T is derived from Eigen::EigenDense<T>
        /// @see https://stackoverflow.com/a/51472601/5253097
        template<class T>
        constexpr bool is_eigen_dense = is_eigen_dense_f(reinterpret_cast<T*>(NULL));

        /// True if T is derived from Eigen::EigenMatrix<T>
        /// @see https://stackoverflow.com/a/51472601/5253097
        template<class T>
        constexpr bool is_eigen_matrix = is_eigen_matrix_f(reinterpret_cast<T*>(NULL));

        /// True if T is derived from Eigen::EigenBlock
        /// @see https://stackoverflow.com/a/51472601/5253097
        template<class T>
        constexpr bool is_eigen_block = is_eigen_block_f(reinterpret_cast<T*>(NULL));

        /// True if T is derived from Eigen::MapBase
        /// @see https://stackoverflow.com/a/51472601/5253097
        template<class T>
        constexpr bool is_eigen_map = is_eigen_map_f(reinterpret_cast<T*>(NULL));
    }

    /// True if T is derived from Eigen::EigenBase
    /// @see https://stackoverflow.com/a/51472601/5253097
    template<class T>
    constexpr bool is_eigen_type = internal::is_eigen_type_f(reinterpret_cast<T*>(NULL));

    // -----------------------------------------------------------------------------
    // Data traits

    namespace internal
    {
        /// Layout for Eigen::Dense types that are not derived from Eigen::Block
        /// nor from Eigen::Map
        template< class matrix_t >
        struct LayoutImpl< matrix_t, typename std::enable_if<
            is_eigen_dense<matrix_t> &&
            !is_eigen_block<matrix_t> &&
            !is_eigen_map<matrix_t>
        ,int>::type >
        {
            static constexpr Layout layout = (matrix_t::IsRowMajor)
                ? Layout::RowMajor
                : Layout::ColMajor;
        };

        /// Layout for Eigen::Block
        template<class XprType, int BlockRows, int BlockCols, bool InnerPanel>
        struct LayoutImpl< Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>, int >
        {
            static constexpr Layout layout = layout<XprType>;
        };

        /// Layout for const Eigen::Block
        template<class XprType, int BlockRows, int BlockCols, bool InnerPanel>
        struct LayoutImpl< const Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>, int >
        {
            static constexpr Layout layout = layout<XprType>;
        };

        /// Layout for Eigen::Map
        template<class PlainObjectType, int MapOptions, class StrideType>
        struct LayoutImpl< Eigen::Map<PlainObjectType,MapOptions,StrideType>, int >
        {
            static constexpr Layout layout = layout<PlainObjectType>;
        };

        /// Layout for const Eigen::Map
        template<class PlainObjectType, int MapOptions, class StrideType>
        struct LayoutImpl< const Eigen::Map<PlainObjectType,MapOptions,StrideType>, int >
        {
            static constexpr Layout layout = layout<PlainObjectType>;
        };

        /// Transpose for Eigen::Matrix
        template< class matrix_t >
        struct TransposeTypeImpl< matrix_t, typename std::enable_if<
            is_eigen_matrix<matrix_t>
        ,int>::type >
        {
            using type = Eigen::Matrix<
                typename matrix_t::Scalar,
                matrix_t::ColsAtCompileTime,
                matrix_t::RowsAtCompileTime, 
                (matrix_t::IsRowMajor) ? Eigen::RowMajor : Eigen::ColMajor,
                matrix_t::MaxColsAtCompileTime,
                matrix_t::MaxRowsAtCompileTime
            >;
        };

        /// Create Eigen::Matrix, @see Create
        template<typename U>
        struct CreateImpl< U, typename std::enable_if< is_eigen_dense<U> ,int>::type >
        {
            using T = typename U::Scalar;
            static constexpr int Rows_ = (U::RowsAtCompileTime == 1) ? 1 : Eigen::Dynamic;
            static constexpr int Cols_ = (U::ColsAtCompileTime == 1) ? 1 : Eigen::Dynamic;
            static constexpr int Options_ = (U::IsRowMajor) ? Eigen::RowMajor : Eigen::ColMajor;

            using matrix_t  = Eigen::Matrix< T, Rows_, Cols_, Options_ >;
            using idx_t     = Eigen::Index;
            using map_t     = Eigen::Map< matrix_t, Eigen::Unaligned, Eigen::OuterStride<> >;

            inline constexpr auto
            operator()( std::vector<T>& v, idx_t m, idx_t n = 1 ) const {
                assert( m >= 0 && n >= 0 );
                v.resize(0);
                return matrix_t( m, n );
            }

            inline constexpr auto
            operator()( const Workspace& W, idx_t m, idx_t n, Workspace& rW ) const {
            
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
    }

    // -----------------------------------------------------------------------------
    // Data descriptors for Eigen datatypes

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
    /* 
     * It is important to separate block operations between: 
     * 1. Data types that do not derive from Eigen::Block
     * 2. Data types that derive from Eigen::Block
     * 
     * This is done to avoid the creation of classes like:
     * Block< Block< Block< MaxtrixXd > > >
     * which would increase the number of template specializations in <T>LAPACK.
     * 
     * Moreover, recursive algorithms call themselves using blocks of thier
     * matrices. This creates a dead lock at compile time if the sizes are
     * given at runtime, and the compiler cannot deduce where to stop the
     * recursion for the template specialization.
     */

    #define isSlice(SliceSpec) !std::is_convertible< SliceSpec, Eigen::Index >::value

    // Block operations for Eigen::Dense that are not derived from Eigen::Block

    template< class T, class SliceSpecRow, class SliceSpecCol, typename std::enable_if<
        isSlice(SliceSpecRow) && isSlice(SliceSpecCol)
        && internal::is_eigen_dense<T> && !internal::is_eigen_block<T>
    , int >::type = 0 >
    inline constexpr auto
    slice( T& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        return A.block( rows.first, cols.first, rows.second-rows.first, cols.second-cols.first );
    }

    template<class T, typename SliceSpecCol, typename std::enable_if<
        internal::is_eigen_dense<T> && !internal::is_eigen_block<T>
    , int >::type = 0 >
    inline constexpr auto
    slice( T& A, Eigen::Index rowIdx, SliceSpecCol&& cols ) noexcept
    {
        return A.row( rowIdx ).segment( cols.first, cols.second-cols.first );
    }

    template<class T, typename SliceSpecRow, typename std::enable_if<
        internal::is_eigen_dense<T> && !internal::is_eigen_block<T>
    , int >::type = 0 >
    inline constexpr auto
    slice( T& A, SliceSpecRow&& rows, Eigen::Index colIdx ) noexcept
    {
        return A.col( colIdx ).segment( rows.first, rows.second-rows.first );
    }

    template<class T, typename SliceSpec, typename std::enable_if<
        internal::is_eigen_dense<T> && !internal::is_eigen_block<T>
    , int >::type = 0 >
    inline constexpr auto
    slice( T& x, SliceSpec&& range ) noexcept
    {
        return x.segment( range.first, range.second-range.first );
    }

    template<class T, typename SliceSpec, typename std::enable_if<
        internal::is_eigen_dense<T> && !internal::is_eigen_block<T>
    , int >::type = 0 >
    inline constexpr auto
    rows( T& A, SliceSpec&& rows ) noexcept
    {
        return A.middleRows( rows.first, rows.second-rows.first );
    }

    template<class T, typename std::enable_if<
        internal::is_eigen_dense<T> && !internal::is_eigen_block<T>
    , int >::type = 0 >
    inline constexpr auto row( T& A, Eigen::Index rowIdx ) noexcept
    {
        return A.row( rowIdx );
    }

    template<class T, typename SliceSpec, typename std::enable_if<
        internal::is_eigen_dense<T> && !internal::is_eigen_block<T>
    , int >::type = 0 >
    inline constexpr auto cols( T& A, SliceSpec&& cols ) noexcept
    {
        return A.middleCols( cols.first, cols.second-cols.first );
    }

    template<class T, typename std::enable_if<
        internal::is_eigen_dense<T> && !internal::is_eigen_block<T>
    , int >::type = 0 >
    inline constexpr auto col( T& A, Eigen::Index colIdx ) noexcept
    {
        return A.col( colIdx );
    }

    // Block operations for Eigen::Dense that are derived from Eigen::Block

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

    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpec>
    inline constexpr auto
    slice( Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& x, SliceSpec&& range ) noexcept
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

    template< class XprType, int BlockRows, int BlockCols, bool InnerPanel, class SliceSpecRow, class SliceSpecCol,
              typename std::enable_if< isSlice(SliceSpecRow) && isSlice(SliceSpecCol), int >::type = 0 >
    inline constexpr auto
    slice( const Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, SliceSpecRow&& rows, SliceSpecCol&& cols ) noexcept
    {
        assert( rows.second <= A.rows() );
        assert( cols.second <= A.cols() );

        return Eigen::Block< const XprType, Eigen::Dynamic, Eigen::Dynamic, InnerPanel >
        (   A.nestedExpression(),
            A.startRow() + rows.first, A.startCol() + cols.first,
            rows.second-rows.first, cols.second-cols.first );
    }

    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpecCol>
    inline constexpr auto
    slice( const Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, Eigen::Index rowIdx, SliceSpecCol&& cols ) noexcept
    {
        assert( rowIdx < A.rows() );
        assert( cols.second <= A.cols() );

        return Eigen::Block< const XprType, 1, Eigen::Dynamic, InnerPanel >
        (   A.nestedExpression(),
            A.startRow() + rowIdx, A.startCol() + cols.first,
            1, cols.second-cols.first );
    }

    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpecRow>
    inline constexpr auto
    slice( const Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, SliceSpecRow&& rows, Eigen::Index colIdx ) noexcept
    {
        assert( rows.second <= A.rows() );
        assert( colIdx < A.cols() );

        return Eigen::Block< const XprType, Eigen::Dynamic, 1, InnerPanel >
        (   A.nestedExpression(),
            A.startRow() + rows.first, A.startCol() + colIdx,
            rows.second-rows.first, 1 );
    }

    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpec>
    inline constexpr auto
    slice( const Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& x, SliceSpec&& range ) noexcept
    {
        return x.segment( range.first, range.second-range.first );
    }

    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpec>
    inline constexpr auto
    rows( const Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, SliceSpec&& rows ) noexcept
    {
        assert( rows.second <= A.rows() );

        return Eigen::Block< const XprType, Eigen::Dynamic, BlockCols, InnerPanel >
        (   A.nestedExpression(),
            A.startRow() + rows.first, A.startCol(),
            rows.second-rows.first, A.cols() );
    }

    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel>
    inline constexpr auto
    row( const Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, Eigen::Index rowIdx ) noexcept
    {
        assert( rowIdx < A.rows() );

        return Eigen::Block< const XprType, 1, BlockCols, InnerPanel >
        (   A.nestedExpression(),
            A.startRow() + rowIdx, A.startCol(),
            1, A.cols() );
    }

    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel, typename SliceSpec>
    inline constexpr auto
    cols( const Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, SliceSpec&& cols ) noexcept
    {
        assert( cols.second <= A.cols() );

        return Eigen::Block< const XprType, BlockRows, Eigen::Dynamic, InnerPanel >
        (   A.nestedExpression(),
            A.startRow(), A.startCol() + cols.first,
            A.rows(), cols.second-cols.first );
    }
    
    template<class XprType, int BlockRows, int BlockCols, bool InnerPanel>
    inline constexpr auto
    col( const Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& A, Eigen::Index colIdx ) noexcept
    {
        assert( colIdx < A.cols() );

        return Eigen::Block< const XprType, BlockRows, 1, InnerPanel >
        (   A.nestedExpression(),
            A.startRow(), A.startCol() + colIdx,
            A.rows(), 1 );
    }

    #undef isSlice

    /// Get the Diagonal of an Eigen Matrix
    template<class T, typename std::enable_if< internal::is_eigen_matrix<T> , int >::type = 0 >
    inline constexpr auto diag( T& A, int diagIdx = 0 ) noexcept
    {
        return A.diagonal( diagIdx );
    }

    // -----------------------------------------------------------------------------
    // Deduce matrix and vector type from two provided ones:

    namespace internal {

        #ifdef TLAPACK_PREFERRED_MATRIX_EIGEN

            #ifndef TLAPACK_LEGACYARRAY_HH
                #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) true
            #else
                #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) !is_legacy_type<T>
            #endif

            // for two types
            // should be especialized for every new matrix class
            template< typename matrixA_t, typename matrixB_t >
            struct matrix_type_traits< matrixA_t, matrixB_t, typename std::enable_if<
                TLAPACK_USE_PREFERRED_MATRIX_TYPE(matrixA_t) ||
                TLAPACK_USE_PREFERRED_MATRIX_TYPE(matrixB_t)
            ,int>::type >
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
            struct vector_type_traits< vecA_t, vecB_t, typename std::enable_if<
                TLAPACK_USE_PREFERRED_MATRIX_TYPE(vecA_t) ||
                TLAPACK_USE_PREFERRED_MATRIX_TYPE(vecB_t)
            ,int>::type >
            {
                using T = scalar_type< type_t<vecA_t>, type_t<vecB_t> >;
                using type = Eigen::Matrix< T, Eigen::Dynamic, 1 >;
            };

        #else

            template< class matrixA_t, class matrixB_t >
            struct matrix_type_traits< matrixA_t, matrixB_t, typename std::enable_if<
                is_eigen_dense<matrixA_t> && is_eigen_dense<matrixB_t>
            ,int>::type >
            {
                using T = scalar_type< typename matrixA_t::Scalar, typename matrixB_t::Scalar >;
                static constexpr int Options_ = (matrixA_t::IsRowMajor && matrixB_t::IsRowMajor)
                    ? Eigen::RowMajor
                    : Eigen::ColMajor;

                using type = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Options_ >;
            };
            
            template< typename vecA_t, typename vecB_t >
            struct vector_type_traits< vecA_t, vecB_t, typename std::enable_if<
                is_eigen_dense<vecA_t> && is_eigen_dense<vecB_t>
            ,int>::type >
            {
                using T = scalar_type< typename vecA_t::Scalar, typename vecB_t::Scalar >;
                using type = Eigen::Matrix< T, Eigen::Dynamic, 1 >;
            };

        #endif // TLAPACK_PREFERRED_MATRIX
    }

    // -----------------------------------------------------------------------------
    // Cast to Legacy arrays

    template< class T, int Rows, int Cols, int Options, int MaxRows, int MaxCols >
    inline constexpr auto
    legacy_matrix( const Eigen::Matrix< T, Rows, Cols, Options, MaxRows, MaxCols >& A ) noexcept
    {
        using matrix_t = Eigen::Matrix< T, Rows, Cols, Options, MaxRows, MaxCols >;
        using idx_t = Eigen::Index;
        constexpr Layout L = layout<matrix_t>;

        return legacyMatrix<T,idx_t,L>( A.rows(), A.cols(), (T*) A.data() );
    }

    template< class Derived >
    inline constexpr auto
    legacy_matrix( const Eigen::MapBase<Derived,Eigen::ReadOnlyAccessors>& A ) noexcept
    {
        using T = typename Derived::Scalar;
        using idx_t = Eigen::Index;
        constexpr Layout L = layout<Derived>;
        assert( A.innerStride() == 1 || A.innerSize() <= 1 || A.outerStride() == 1 || A.outerSize() <= 1 );

        return legacyMatrix<T,idx_t,L>( A.rows(), A.cols(), (T*) A.data(), A.outerStride() );
    }

    template< class T, int Rows, int Cols, int Options, int MaxRows, int MaxCols >
    inline constexpr auto
    legacy_vector( const Eigen::Matrix< T, Rows, Cols, Options, MaxRows, MaxCols >& A ) noexcept
    {
        using idx_t = Eigen::Index;
        assert( A.innerSize() <= 1 || A.outerSize() <= 1 );
        
        return legacyVector<T,idx_t,idx_t>( A.size(), (T*) A.data(),
            (A.outerSize() == 1) ? A.innerStride() : A.outerStride() );
    }

    template< class Derived >
    inline constexpr auto
    legacy_vector( const Eigen::MapBase<Derived,Eigen::ReadOnlyAccessors>& A ) noexcept
    {
        using T = typename Derived::Scalar;
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
