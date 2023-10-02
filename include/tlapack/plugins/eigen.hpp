/// @file eigen.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_EIGEN_HH
#define TLAPACK_EIGEN_HH

#include <Eigen/Core>
#include <cassert>

#include "tlapack/base/arrayTraits.hpp"

namespace tlapack {

// -----------------------------------------------------------------------------
// Helpers

namespace eigen {
    namespace internal {
        // Auxiliary constexpr routines

        template <class Derived>
        std::true_type is_eigen_dense_f(const Eigen::DenseBase<Derived>*);
        std::false_type is_eigen_dense_f(const void*);

        template <class Derived>
        std::true_type is_eigen_matrix_f(const Eigen::MatrixBase<Derived>*);
        std::false_type is_eigen_matrix_f(const void*);

        template <class XprType, int BlockRows, int BlockCols, bool InnerPanel>
        std::true_type is_eigen_block_f(
            const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>*);
        std::false_type is_eigen_block_f(const void*);

        /// True if T is derived from Eigen::EigenDense<T>
        /// @see https://stackoverflow.com/a/25223400/5253097
        template <class T>
        constexpr bool is_eigen_dense =
            decltype(is_eigen_dense_f(std::declval<T*>()))::value;

        /// True if T is derived from Eigen::EigenMatrix<T>
        /// @see https://stackoverflow.com/a/25223400/5253097
        template <class T>
        constexpr bool is_eigen_matrix =
            decltype(is_eigen_matrix_f(std::declval<T*>()))::value;

        /// True if T is derived from Eigen::EigenBlock
        /// @see https://stackoverflow.com/a/25223400/5253097
        template <class T>
        constexpr bool is_eigen_block =
            decltype(is_eigen_block_f(std::declval<T*>()))::value;

        template <class>
        struct isStdComplex : public std::false_type {};
        template <class T>
        struct isStdComplex<std::complex<T>> : public std::true_type {};

    }  // namespace internal

    /// True if T is derived from Eigen::DenseBase
    /// @see https://stackoverflow.com/a/25223400/5253097
    template <class T>
    constexpr bool is_eigen_type = internal::is_eigen_dense<T>;

}  // namespace eigen

// -----------------------------------------------------------------------------
// Data traits

namespace traits {

    /// Layout for Eigen::Dense types
    template <class matrix_t>
    struct layout_trait<
        matrix_t,
        typename std::enable_if<eigen::is_eigen_type<matrix_t> &&
                                    (matrix_t::InnerStrideAtCompileTime == 1 ||
                                     matrix_t::OuterStrideAtCompileTime == 1),
                                int>::type> {
        static constexpr Layout value =
            (matrix_t::IsVectorAtCompileTime)
                ? Layout::Strided
                : ((matrix_t::IsRowMajor) ? Layout::RowMajor
                                          : Layout::ColMajor);
    };

    template <class matrix_t>
    struct real_type_traits<
        matrix_t,
        typename std::enable_if<eigen::is_eigen_type<matrix_t>, int>::type> {
        using type = Eigen::Matrix<real_type<typename matrix_t::Scalar>,
                                   matrix_t::RowsAtCompileTime,
                                   matrix_t::ColsAtCompileTime,
                                   matrix_t::IsRowMajor ? Eigen::RowMajor
                                                        : Eigen::ColMajor,
                                   matrix_t::MaxRowsAtCompileTime,
                                   matrix_t::MaxColsAtCompileTime>;
    };

    template <class matrix_t>
    struct complex_type_traits<
        matrix_t,
        typename std::enable_if<eigen::is_eigen_type<matrix_t>, int>::type> {
        using type = Eigen::Matrix<complex_type<typename matrix_t::Scalar>,
                                   matrix_t::RowsAtCompileTime,
                                   matrix_t::ColsAtCompileTime,
                                   matrix_t::IsRowMajor ? Eigen::RowMajor
                                                        : Eigen::ColMajor,
                                   matrix_t::MaxRowsAtCompileTime,
                                   matrix_t::MaxColsAtCompileTime>;
    };

    /// Create Eigen::Matrix, @see Create
    template <typename U>
    struct CreateFunctor<
        U,
        typename std::enable_if<eigen::is_eigen_type<U>, int>::type> {
        static constexpr int Rows_ =
            (U::RowsAtCompileTime == 1) ? 1 : Eigen::Dynamic;
        static constexpr int Cols_ =
            (U::ColsAtCompileTime == 1) ? 1 : Eigen::Dynamic;
        static constexpr int Options_ =
            (U::IsRowMajor) ? Eigen::RowMajor : Eigen::ColMajor;

        template <typename T>
        constexpr auto operator()(std::vector<T>& v,
                                  Eigen::Index m,
                                  Eigen::Index n = 1) const
        {
            assert(m >= 0 && n >= 0);
            v.resize(0);
            return Eigen::Matrix<T, Rows_, Cols_, Options_>(m, n);
        }
    };
}  // namespace traits

// -----------------------------------------------------------------------------
// Data descriptors for Eigen datatypes

// Size
template <
    class T,
    int Rows,
    int Cols,
    int Options,
    int MaxRows,
    int MaxCols
#if __cplusplus >= 201703L
    // Avoids conflict with std::size
    ,
    std::enable_if_t<!(eigen::internal::isStdComplex<std::decay_t<T>>::value),
                     int> = 0
#endif
    >
constexpr auto size(
    const Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>& x) noexcept
{
    return x.size();
}
template <class Derived>
constexpr auto size(const Eigen::EigenBase<Derived>& x) noexcept
{
    return x.size();
}
// Number of rows
template <class T>
constexpr auto nrows(const Eigen::EigenBase<T>& x) noexcept
{
    return x.rows();
}
// Number of columns
template <class T>
constexpr auto ncols(const Eigen::EigenBase<T>& x) noexcept
{
    return x.cols();
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

#define isSlice(SliceSpec) !std::is_convertible<SliceSpec, Eigen::Index>::value

// Block operations for Eigen::Dense that are not derived from Eigen::Block

template <
    class T,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol) &&
                                eigen::is_eigen_type<T> &&
                                !eigen::internal::is_eigen_block<T>,
                            int>::type = 0>
constexpr auto slice(T& A, SliceSpecRow&& rows, SliceSpecCol&& cols) noexcept
{
    return A.block(rows.first, cols.first, rows.second - rows.first,
                   cols.second - cols.first);
}

template <class T,
          typename SliceSpecCol,
          typename std::enable_if<eigen::is_eigen_type<T> &&
                                      !eigen::internal::is_eigen_block<T>,
                                  int>::type = 0>
constexpr auto slice(T& A, Eigen::Index rowIdx, SliceSpecCol&& cols) noexcept
{
    return A.row(rowIdx).segment(cols.first, cols.second - cols.first);
}

template <class T,
          typename SliceSpecRow,
          typename std::enable_if<eigen::is_eigen_type<T> &&
                                      !eigen::internal::is_eigen_block<T>,
                                  int>::type = 0>
constexpr auto slice(T& A, SliceSpecRow&& rows, Eigen::Index colIdx) noexcept
{
    return A.col(colIdx).segment(rows.first, rows.second - rows.first);
}

template <class T,
          typename SliceSpec,
          typename std::enable_if<eigen::is_eigen_type<T> &&
                                      !eigen::internal::is_eigen_block<T>,
                                  int>::type = 0>
constexpr auto slice(T& x, SliceSpec&& range) noexcept
{
    return x.segment(range.first, range.second - range.first);
}

template <class T,
          typename SliceSpec,
          typename std::enable_if<eigen::is_eigen_type<T> &&
                                      !eigen::internal::is_eigen_block<T>,
                                  int>::type = 0>
constexpr auto rows(T& A, SliceSpec&& rows) noexcept
{
    return A.middleRows(rows.first, rows.second - rows.first);
}

template <class T,
          typename std::enable_if<eigen::is_eigen_type<T> &&
                                      !eigen::internal::is_eigen_block<T>,
                                  int>::type = 0>
constexpr auto row(T& A, Eigen::Index rowIdx) noexcept
{
    return A.row(rowIdx);
}

template <class T,
          typename SliceSpec,
          typename std::enable_if<eigen::is_eigen_type<T> &&
                                      !eigen::internal::is_eigen_block<T>,
                                  int>::type = 0>
constexpr auto cols(T& A, SliceSpec&& cols) noexcept
{
    return A.middleCols(cols.first, cols.second - cols.first);
}

template <class T,
          typename std::enable_if<eigen::is_eigen_type<T> &&
                                      !eigen::internal::is_eigen_block<T>,
                                  int>::type = 0>
constexpr auto col(T& A, Eigen::Index colIdx) noexcept
{
    return A.col(colIdx);
}

// Block operations for Eigen::Dense that are derived from Eigen::Block

template <
    class XprType,
    int BlockRows,
    int BlockCols,
    bool InnerPanel,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
                            int>::type = 0>
constexpr auto slice(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
                     SliceSpecRow&& rows,
                     SliceSpecCol&& cols) noexcept
{
    assert(rows.second <= A.rows());
    assert(cols.second <= A.cols());

    return Eigen::Block<XprType, Eigen::Dynamic, Eigen::Dynamic, InnerPanel>(
        A.nestedExpression(), A.startRow() + rows.first,
        A.startCol() + cols.first, rows.second - rows.first,
        cols.second - cols.first);
}

template <class XprType,
          int BlockRows,
          int BlockCols,
          bool InnerPanel,
          typename SliceSpecCol>
constexpr auto slice(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
                     Eigen::Index rowIdx,
                     SliceSpecCol&& cols) noexcept
{
    assert(rowIdx < A.rows());
    assert(cols.second <= A.cols());

    return Eigen::Block<XprType, 1, Eigen::Dynamic, InnerPanel>(
        A.nestedExpression(), A.startRow() + rowIdx, A.startCol() + cols.first,
        1, cols.second - cols.first);
}

template <class XprType,
          int BlockRows,
          int BlockCols,
          bool InnerPanel,
          typename SliceSpecRow>
constexpr auto slice(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
                     SliceSpecRow&& rows,
                     Eigen::Index colIdx) noexcept
{
    assert(rows.second <= A.rows());
    assert(colIdx < A.cols());

    return Eigen::Block<XprType, Eigen::Dynamic, 1, InnerPanel>(
        A.nestedExpression(), A.startRow() + rows.first, A.startCol() + colIdx,
        rows.second - rows.first, 1);
}

template <class XprType, int BlockRows, bool InnerPanel, typename SliceSpec>
constexpr auto slice(Eigen::Block<XprType, BlockRows, 1, InnerPanel>& x,
                     SliceSpec&& range) noexcept
{
    assert(range.second <= x.size());

    return Eigen::Block<XprType, Eigen::Dynamic, 1, InnerPanel>(
        x.nestedExpression(), x.startRow() + range.first, x.startCol(),
        range.second - range.first, 1);
}

template <class XprType, int BlockCols, bool InnerPanel, typename SliceSpec>
constexpr auto slice(Eigen::Block<XprType, 1, BlockCols, InnerPanel>& x,
                     SliceSpec&& range) noexcept
{
    assert(range.second <= x.size());

    return Eigen::Block<XprType, 1, Eigen::Dynamic, InnerPanel>(
        x.nestedExpression(), x.startRow(), x.startCol() + range.first, 1,
        range.second - range.first);
}

template <class XprType,
          int BlockRows,
          int BlockCols,
          bool InnerPanel,
          typename SliceSpec>
constexpr auto rows(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
                    SliceSpec&& rows) noexcept
{
    assert(rows.second <= A.rows());

    return Eigen::Block<XprType, Eigen::Dynamic, BlockCols, InnerPanel>(
        A.nestedExpression(), A.startRow() + rows.first, A.startCol(),
        rows.second - rows.first, A.cols());
}

template <class XprType, int BlockRows, int BlockCols, bool InnerPanel>
constexpr auto row(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
                   Eigen::Index rowIdx) noexcept
{
    assert(rowIdx < A.rows());

    return Eigen::Block<XprType, 1, BlockCols, InnerPanel>(
        A.nestedExpression(), A.startRow() + rowIdx, A.startCol(), 1, A.cols());
}

template <class XprType,
          int BlockRows,
          int BlockCols,
          bool InnerPanel,
          typename SliceSpec>
constexpr auto cols(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
                    SliceSpec&& cols) noexcept
{
    assert(cols.second <= A.cols());

    return Eigen::Block<XprType, BlockRows, Eigen::Dynamic, InnerPanel>(
        A.nestedExpression(), A.startRow(), A.startCol() + cols.first, A.rows(),
        cols.second - cols.first);
}

template <class XprType, int BlockRows, int BlockCols, bool InnerPanel>
constexpr auto col(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
                   Eigen::Index colIdx) noexcept
{
    assert(colIdx < A.cols());

    return Eigen::Block<XprType, BlockRows, 1, InnerPanel>(
        A.nestedExpression(), A.startRow(), A.startCol() + colIdx, A.rows(), 1);
}

template <
    class XprType,
    int BlockRows,
    int BlockCols,
    bool InnerPanel,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
                            int>::type = 0>
constexpr auto slice(
    const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
    SliceSpecRow&& rows,
    SliceSpecCol&& cols) noexcept
{
    assert(rows.second <= A.rows());
    assert(cols.second <= A.cols());

    return Eigen::Block<const XprType, Eigen::Dynamic, Eigen::Dynamic,
                        InnerPanel>(
        A.nestedExpression(), A.startRow() + rows.first,
        A.startCol() + cols.first, rows.second - rows.first,
        cols.second - cols.first);
}

template <class XprType,
          int BlockRows,
          int BlockCols,
          bool InnerPanel,
          typename SliceSpecCol>
constexpr auto slice(
    const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
    Eigen::Index rowIdx,
    SliceSpecCol&& cols) noexcept
{
    assert(rowIdx < A.rows());
    assert(cols.second <= A.cols());

    return Eigen::Block<const XprType, 1, Eigen::Dynamic, InnerPanel>(
        A.nestedExpression(), A.startRow() + rowIdx, A.startCol() + cols.first,
        1, cols.second - cols.first);
}

template <class XprType,
          int BlockRows,
          int BlockCols,
          bool InnerPanel,
          typename SliceSpecRow>
constexpr auto slice(
    const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
    SliceSpecRow&& rows,
    Eigen::Index colIdx) noexcept
{
    assert(rows.second <= A.rows());
    assert(colIdx < A.cols());

    return Eigen::Block<const XprType, Eigen::Dynamic, 1, InnerPanel>(
        A.nestedExpression(), A.startRow() + rows.first, A.startCol() + colIdx,
        rows.second - rows.first, 1);
}

template <class XprType, int BlockRows, bool InnerPanel, typename SliceSpec>
constexpr auto slice(const Eigen::Block<XprType, BlockRows, 1, InnerPanel>& x,
                     SliceSpec&& range) noexcept
{
    assert(range.second <= x.size());

    return Eigen::Block<const XprType, Eigen::Dynamic, 1, InnerPanel>(
        x.nestedExpression(), x.startRow() + range.first, x.startCol(),
        range.second - range.first, 1);
}

template <class XprType, int BlockCols, bool InnerPanel, typename SliceSpec>
constexpr auto slice(const Eigen::Block<XprType, 1, BlockCols, InnerPanel>& x,
                     SliceSpec&& range) noexcept
{
    assert(range.second <= x.size());

    return Eigen::Block<const XprType, 1, Eigen::Dynamic, InnerPanel>(
        x.nestedExpression(), x.startRow(), x.startCol() + range.first, 1,
        range.second - range.first);
}

template <class XprType,
          int BlockRows,
          int BlockCols,
          bool InnerPanel,
          typename SliceSpec>
constexpr auto rows(
    const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
    SliceSpec&& rows) noexcept
{
    assert(rows.second <= A.rows());

    return Eigen::Block<const XprType, Eigen::Dynamic, BlockCols, InnerPanel>(
        A.nestedExpression(), A.startRow() + rows.first, A.startCol(),
        rows.second - rows.first, A.cols());
}

template <class XprType, int BlockRows, int BlockCols, bool InnerPanel>
constexpr auto row(
    const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
    Eigen::Index rowIdx) noexcept
{
    assert(rowIdx < A.rows());

    return Eigen::Block<const XprType, 1, BlockCols, InnerPanel>(
        A.nestedExpression(), A.startRow() + rowIdx, A.startCol(), 1, A.cols());
}

template <class XprType,
          int BlockRows,
          int BlockCols,
          bool InnerPanel,
          typename SliceSpec>
constexpr auto cols(
    const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
    SliceSpec&& cols) noexcept
{
    assert(cols.second <= A.cols());

    return Eigen::Block<const XprType, BlockRows, Eigen::Dynamic, InnerPanel>(
        A.nestedExpression(), A.startRow(), A.startCol() + cols.first, A.rows(),
        cols.second - cols.first);
}

template <class XprType, int BlockRows, int BlockCols, bool InnerPanel>
constexpr auto col(
    const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& A,
    Eigen::Index colIdx) noexcept
{
    assert(colIdx < A.cols());

    return Eigen::Block<const XprType, BlockRows, 1, InnerPanel>(
        A.nestedExpression(), A.startRow(), A.startCol() + colIdx, A.rows(), 1);
}

#undef isSlice

/// Get the Diagonal of an Eigen Matrix
template <
    class T,
    typename std::enable_if<eigen::internal::is_eigen_matrix<T>, int>::type = 0>
constexpr auto diag(T& A, int diagIdx = 0) noexcept
{
    return A.diagonal(diagIdx);
}

// Transpose view
template <class matrix_t,
          typename std::enable_if<(eigen::is_eigen_type<matrix_t> &&
                                   matrix_t::IsVectorAtCompileTime),
                                  int>::type = 0>
constexpr auto transpose_view(matrix_t& A) noexcept
{
    using T = typename matrix_t::Scalar;
    using Stride = Eigen::InnerStride<>;

    constexpr int Rows_ = matrix_t::ColsAtCompileTime;
    constexpr int Cols_ = matrix_t::RowsAtCompileTime;

    using transpose_t = Eigen::Matrix<
        T, Rows_, Cols_,
        (matrix_t::IsRowMajor) ? Eigen::ColMajor : Eigen::RowMajor,
        matrix_t::MaxColsAtCompileTime, matrix_t::MaxRowsAtCompileTime>;

    using map_t = Eigen::Map<transpose_t, Eigen::Unaligned, Stride>;

    return map_t((T*)A.data(), A.size(), A.innerStride());
}
template <class matrix_t,
          typename std::enable_if<(eigen::is_eigen_type<matrix_t> &&
                                   !matrix_t::IsVectorAtCompileTime),
                                  int>::type = 0>
constexpr auto transpose_view(matrix_t& A) noexcept
{
    using T = typename matrix_t::Scalar;
    using Stride = Eigen::OuterStride<>;
    assert(A.innerStride() == 1);

    constexpr int Rows_ = matrix_t::ColsAtCompileTime;
    constexpr int Cols_ = matrix_t::RowsAtCompileTime;

    using transpose_t = Eigen::Matrix<
        T, Rows_, Cols_,
        (matrix_t::IsRowMajor) ? Eigen::ColMajor : Eigen::RowMajor,
        matrix_t::MaxColsAtCompileTime, matrix_t::MaxRowsAtCompileTime>;

    using map_t = Eigen::Map<transpose_t, Eigen::Unaligned, Stride>;

    return map_t((T*)A.data(), A.cols(), A.rows(), A.outerStride());
}

template <
    class matrix_t,
    typename std::enable_if<eigen::is_eigen_type<matrix_t>, int>::type = 0>
auto reshape(matrix_t& A, Eigen::Index m, Eigen::Index n) noexcept
{
    using T = typename matrix_t::Scalar;
    using Stride = Eigen::OuterStride<>;
    assert(A.innerStride() == 1);

    using rmatrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                    (matrix_t::IsRowMajor) ? Eigen::RowMajor
                                                           : Eigen::ColMajor>;
    using map_t = Eigen::Map<rmatrix_t, Eigen::Unaligned, Stride>;

    if (m == A.rows() && n == A.cols())
        return map_t((T*)A.data(), m, n, A.outerStride());
    else {
        assert((m * n == A.size()) &&
               "reshape: new shape must have the same "
               "number of elements as the original one");
        assert(((!matrix_t::IsRowMajor &&
                 (A.outerStride() == A.rows() || A.cols() <= 1)) ||
                (matrix_t::IsRowMajor &&
                 (A.outerStride() == A.cols() || A.rows() <= 1))) &&
               "reshape: data must be contiguous in memory");
        return map_t((T*)A.data(), m, n, (matrix_t::IsRowMajor ? n : m));
    }
}

// -----------------------------------------------------------------------------
// Deduce matrix and vector type from two provided ones

namespace traits {

#ifdef TLAPACK_PREFERRED_MATRIX_EIGEN

    #ifndef TLAPACK_LEGACY_HH
        #ifndef TLAPACK_MDSPAN_HH
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) true
        #else
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                !mdspan::is_mdspan_type<T>
        #endif
    #else
        #ifndef TLAPACK_MDSPAN_HH
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                !legacy::is_legacy_type<T>
        #else
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                (!legacy::is_legacy_type<T> && !mdspan::is_mdspan_type<T>)
        #endif
    #endif

    // for two types
    // should be especialized for every new matrix class
    template <typename matrixA_t, typename matrixB_t>
    struct matrix_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<TLAPACK_USE_PREFERRED_MATRIX_TYPE(matrixA_t) ||
                                    TLAPACK_USE_PREFERRED_MATRIX_TYPE(
                                        matrixB_t),
                                int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;

        static constexpr Layout LA = layout<matrixA_t>;
        static constexpr Layout LB = layout<matrixB_t>;
        static constexpr int L =
            ((LA == Layout::RowMajor) && (LB == Layout::RowMajor))
                ? Eigen::RowMajor
                : Eigen::ColMajor;

        using type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, L>;
    };

    // for two types
    // should be especialized for every new vector class
    template <typename vecA_t, typename vecB_t>
    struct vector_type_traits<
        vecA_t,
        vecB_t,
        typename std::enable_if<TLAPACK_USE_PREFERRED_MATRIX_TYPE(vecA_t) ||
                                    TLAPACK_USE_PREFERRED_MATRIX_TYPE(vecB_t),
                                int>::type> {
        using T = scalar_type<type_t<vecA_t>, type_t<vecB_t>>;
        using type = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    };

#else

    template <class matrixA_t, class matrixB_t>
    struct matrix_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<eigen::is_eigen_type<matrixA_t> &&
                                    eigen::is_eigen_type<matrixB_t>,
                                int>::type> {
        using T =
            scalar_type<typename matrixA_t::Scalar, typename matrixB_t::Scalar>;
        static constexpr int Options_ =
            (matrixA_t::IsRowMajor && matrixB_t::IsRowMajor) ? Eigen::RowMajor
                                                             : Eigen::ColMajor;

        using type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Options_>;
    };

    template <typename vecA_t, typename vecB_t>
    struct vector_type_traits<
        vecA_t,
        vecB_t,
        typename std::enable_if<eigen::is_eigen_type<vecA_t> &&
                                    eigen::is_eigen_type<vecB_t>,
                                int>::type> {
        using T = scalar_type<typename vecA_t::Scalar, typename vecB_t::Scalar>;
        using type = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    };

#endif  // TLAPACK_PREFERRED_MATRIX
}  // namespace traits

// -----------------------------------------------------------------------------
// Cast to Legacy arrays

template <class T, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
constexpr auto legacy_matrix(
    const Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>& A) noexcept
{
    using matrix_t = Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>;
    using idx_t = Eigen::Index;

    if constexpr (matrix_t::IsVectorAtCompileTime)
        return legacy::Matrix<T, idx_t>{Layout::ColMajor, 1, A.size(),
                                        (T*)A.data(), A.innerStride()};
    else {
        constexpr Layout L = layout<matrix_t>;
        return legacy::Matrix<T, idx_t>{L, A.rows(), A.cols(), (T*)A.data(),
                                        A.outerStride()};
    }
}

template <class Derived>
constexpr auto legacy_matrix(
    const Eigen::MapBase<Derived, Eigen::ReadOnlyAccessors>& A) noexcept
{
    using T = typename Derived::Scalar;
    using idx_t = Eigen::Index;

    if constexpr (Derived::IsVectorAtCompileTime) {
        assert(
            A.outerStride() == 1 ||
            (A.innerStride() == 1 && eigen::internal::is_eigen_block<Derived>));
        return legacy::Matrix<T, idx_t>{Layout::ColMajor, 1, A.size(),
                                        (T*)A.data(), A.innerStride()};
    }
    else {
        assert(A.innerStride() == 1);
        constexpr Layout L = layout<Derived>;
        return legacy::Matrix<T, idx_t>{L, A.rows(), A.cols(), (T*)A.data(),
                                        A.outerStride()};
    }
}

template <class T, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
constexpr auto legacy_vector(
    const Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>& A) noexcept
{
    using matrix_t = Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>;
    using idx_t = Eigen::Index;

    if constexpr (matrix_t::IsVectorAtCompileTime)
        return legacy::Vector<T, idx_t>{A.size(), (T*)A.data(),
                                        A.innerStride()};
    else {
        assert(A.rows() == 1 || A.cols() == 1);
        return legacy::Vector<T, idx_t>{A.size(), (T*)A.data(),
                                        A.outerStride()};
    }
}

template <class Derived>
constexpr auto legacy_vector(
    const Eigen::MapBase<Derived, Eigen::ReadOnlyAccessors>& A) noexcept
{
    using T = typename Derived::Scalar;
    using idx_t = Eigen::Index;

    if constexpr (Derived::IsVectorAtCompileTime) {
        assert(
            A.outerStride() == 1 ||
            (A.innerStride() == 1 && eigen::internal::is_eigen_block<Derived>));
        return legacy::Vector<T, idx_t>{A.size(), (T*)A.data(),
                                        A.innerStride()};
    }
    else {
        assert(A.innerStride() == 1);
        assert(A.rows() == 1 || A.cols() == 1);
        return legacy::Vector<T, idx_t>{A.size(), (T*)A.data(),
                                        A.outerStride()};
    }
}

}  // namespace tlapack

#endif  // TLAPACK_EIGEN_HH
