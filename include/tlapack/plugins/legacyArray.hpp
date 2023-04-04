/// @file legacyArray.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACYARRAY_HH
#define TLAPACK_LEGACYARRAY_HH

#include <cassert>

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/legacyArray.hpp"
#include "tlapack/base/workspace.hpp"

namespace tlapack {

// -----------------------------------------------------------------------------
// Helpers

namespace internal {
    template <typename T, class idx_t, Layout L>
    std::true_type is_legacy_type_f(const legacyMatrix<T, idx_t, L>*);

    template <typename T, class idx_t, class int_t, Direction D>
    std::true_type is_legacy_type_f(const legacyVector<T, idx_t, int_t, D>*);

    std::false_type is_legacy_type_f(const void*);
}  // namespace internal

/// True if T is a legacy array
/// @see https://stackoverflow.com/a/25223400/5253097
template <class T>
constexpr bool is_legacy_type =
    decltype(internal::is_legacy_type_f(std::declval<T*>()))::value;

// -----------------------------------------------------------------------------
// Data traits

namespace internal {
    /// Layout for legacyMatrix
    template <typename T, class idx_t, Layout L>
    struct LayoutImpl<legacyMatrix<T, idx_t, L>, int> {
        static constexpr Layout layout = L;
    };

    /// Layout for legacyVector
    template <typename T, class idx_t, typename int_t, Direction D>
    struct LayoutImpl<legacyVector<T, idx_t, int_t, D>, int> {
        static constexpr Layout layout = Layout::Strided;
    };

    /// Transpose type for legacyMatrix
    template <class T, class idx_t>
    struct TransposeTypeImpl<legacyMatrix<T, idx_t, Layout::ColMajor>, int> {
        using type = legacyMatrix<T, idx_t, Layout::RowMajor>;
    };
    template <class T, class idx_t>
    struct TransposeTypeImpl<legacyMatrix<T, idx_t, Layout::RowMajor>, int> {
        using type = legacyMatrix<T, idx_t, Layout::ColMajor>;
    };

    /// Create legacyMatrix @see Create
    template <class T, class idx_t, Layout layout>
    struct CreateImpl<legacyMatrix<T, idx_t, layout>, int> {
        using matrix_t = legacyMatrix<T, idx_t, layout>;

        inline constexpr auto operator()(std::vector<T>& v,
                                         idx_t m,
                                         idx_t n) const
        {
            assert(m >= 0 && n >= 0);
            v.resize(m * n);  // Allocates space in memory
            return matrix_t(m, n, v.data());
        }

        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         idx_t n,
                                         Workspace& rW) const
        {
            assert(m >= 0 && n >= 0);
            rW = (layout == Layout::ColMajor) ? W.extract(m * sizeof(T), n)
                                              : W.extract(n * sizeof(T), m);
            return (W.isContiguous())
                       ? matrix_t(m, n,
                                  (T*)W.data())  // contiguous space in memory
                       : matrix_t(m, n, (T*)W.data(), W.getLdim() / sizeof(T));
        }

        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         idx_t n) const
        {
            assert(m >= 0 && n >= 0);
            tlapack_check((layout == Layout::ColMajor)
                              ? W.contains(m * sizeof(T), n)
                              : W.contains(n * sizeof(T), m));
            return (W.isContiguous())
                       ? matrix_t(m, n,
                                  (T*)W.data())  // contiguous space in memory
                       : matrix_t(m, n, (T*)W.data(), W.getLdim() / sizeof(T));
        }
    };

    /// Create legacyVector @see Create
    template <class T, class idx_t, typename int_t, Direction D>
    struct CreateImpl<legacyVector<T, idx_t, int_t, D>, int> {
        using vector_t = legacyVector<T, idx_t, int_t, D>;

        inline constexpr auto operator()(std::vector<T>& v, idx_t m) const
        {
            assert(m >= 0);
            v.resize(m);  // Allocates space in memory
            return vector_t(m, v.data());
        }

        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         Workspace& rW) const
        {
            assert(m >= 0);
            rW = W.extract(sizeof(T), m);
            return (W.isContiguous())
                       ? vector_t(m,
                                  (T*)W.data())  // contiguous space in memory
                       : vector_t(m, (T*)W.data(), W.getLdim() / sizeof(T));
        }

        inline constexpr auto operator()(const Workspace& W, idx_t m) const
        {
            assert(m >= 0);
            tlapack_check(W.contains(sizeof(T), m));
            return (W.isContiguous())
                       ? vector_t(m,
                                  (T*)W.data())  // contiguous space in memory
                       : vector_t(m, (T*)W.data(), W.getLdim() / sizeof(T));
        }
    };
}  // namespace internal

// -----------------------------------------------------------------------------
// Data descriptors

// Number of rows of legacyMatrix
template <typename T, class idx_t, Layout layout>
inline constexpr auto nrows(const legacyMatrix<T, idx_t, layout>& A)
{
    return A.m;
}

// Number of columns of legacyMatrix
template <typename T, class idx_t, Layout layout>
inline constexpr auto ncols(const legacyMatrix<T, idx_t, layout>& A)
{
    return A.n;
}

// Size of legacyVector
template <typename T, class idx_t, typename int_t, Direction direction>
inline constexpr auto size(const legacyVector<T, idx_t, int_t, direction>& x)
{
    return x.n;
}

// Number of rows of legacyBandedMatrix
template <typename T, class idx_t>
inline constexpr auto nrows(const legacyBandedMatrix<T, idx_t>& A)
{
    return A.m;
}

// Number of columns of legacyBandedMatrix
template <typename T, class idx_t>
inline constexpr auto ncols(const legacyBandedMatrix<T, idx_t>& A)
{
    return A.n;
}

// Lowerband of legacyBandedMatrix
template <typename T, class idx_t>
inline constexpr auto lowerband(const legacyBandedMatrix<T, idx_t>& A)
{
    return A.kl;
}

// Upperband of legacyBandedMatrix
template <typename T, class idx_t>
inline constexpr auto upperband(const legacyBandedMatrix<T, idx_t>& A)
{
    return A.ku;
}

// -----------------------------------------------------------------------------
// Block operations

#define isSlice(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value

// Slice legacyMatrix
template <
    typename T,
    class idx_t,
    Layout layout,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
                            int>::type = 0>
inline constexpr auto slice(const legacyMatrix<T, idx_t, layout>& A,
                            SliceSpecRow&& rows,
                            SliceSpecCol&& cols) noexcept
{
    assert((rows.first >= 0 and (idx_t) rows.first < nrows(A)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 and (idx_t) rows.second <= nrows(A));
    assert(rows.first <= rows.second);
    assert((cols.first >= 0 and (idx_t) cols.first < ncols(A)) ||
           cols.first == cols.second);
    assert(cols.second >= 0 and (idx_t) cols.second <= ncols(A));
    assert(cols.first <= cols.second);
    return legacyMatrix<T, idx_t, layout>(
        rows.second - rows.first, cols.second - cols.first,
        (layout == Layout::ColMajor) ? &A.ptr[rows.first + cols.first * A.ldim]
                                     : &A.ptr[rows.first * A.ldim + cols.first],
        A.ldim);
}

#undef isSlice

// Slice legacyMatrix over a single row
template <typename T, class idx_t, Layout layout, class SliceSpecCol>
inline constexpr auto slice(const legacyMatrix<T, idx_t, layout>& A,
                            size_type<legacyMatrix<T, idx_t, layout>> rowIdx,
                            SliceSpecCol&& cols) noexcept
{
    assert((cols.first >= 0 and (idx_t) cols.first < ncols(A)) ||
           cols.first == cols.second);
    assert(cols.second >= 0 and (idx_t) cols.second <= ncols(A));
    assert(cols.first <= cols.second);
    assert(rowIdx >= 0 and rowIdx < nrows(A));
    return legacyVector<T, idx_t, idx_t>(
        cols.second - cols.first,
        (layout == Layout::ColMajor) ? &A.ptr[rowIdx + cols.first * A.ldim]
                                     : &A.ptr[rowIdx * A.ldim + cols.first],
        layout == Layout::ColMajor ? A.ldim : 1);
}

// Slice legacyMatrix over a single column
template <typename T, class idx_t, Layout layout, class SliceSpecRow>
inline constexpr auto slice(
    const legacyMatrix<T, idx_t, layout>& A,
    SliceSpecRow&& rows,
    size_type<legacyMatrix<T, idx_t, layout>> colIdx) noexcept
{
    assert((rows.first >= 0 and (idx_t) rows.first < nrows(A)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 and (idx_t) rows.second <= nrows(A));
    assert(rows.first <= rows.second);
    assert(colIdx >= 0 and colIdx < ncols(A));
    return legacyVector<T, idx_t, idx_t>(
        rows.second - rows.first,
        (layout == Layout::ColMajor) ? &A.ptr[rows.first + colIdx * A.ldim]
                                     : &A.ptr[rows.first * A.ldim + colIdx],
        layout == Layout::RowMajor ? A.ldim : 1);
}

// Get rows of legacyMatrix
template <typename T, class idx_t, Layout layout, class SliceSpec>
inline constexpr auto rows(const legacyMatrix<T, idx_t, layout>& A,
                           SliceSpec&& rows) noexcept
{
    assert((rows.first >= 0 and (idx_t) rows.first < nrows(A)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 and (idx_t) rows.second <= nrows(A));
    assert(rows.first <= rows.second);
    return legacyMatrix<T, idx_t, layout>(rows.second - rows.first, A.n,
                                          (layout == Layout::ColMajor)
                                              ? &A.ptr[rows.first]
                                              : &A.ptr[rows.first * A.ldim],
                                          A.ldim);
}

// Get a row of a column-major legacyMatrix
template <typename T, class idx_t>
inline constexpr auto row(const legacyMatrix<T, idx_t>& A,
                          size_type<legacyMatrix<T, idx_t>> rowIdx) noexcept
{
    assert(rowIdx >= 0 and rowIdx < nrows(A));
    return legacyVector<T, idx_t, idx_t>(A.n, &A.ptr[rowIdx], A.ldim);
}

// Get a row of a row-major legacyMatrix
template <typename T, class idx_t>
inline constexpr auto row(
    const legacyMatrix<T, idx_t, Layout::RowMajor>& A,
    size_type<legacyMatrix<T, idx_t, Layout::RowMajor>> rowIdx) noexcept
{
    assert(rowIdx >= 0 and rowIdx < nrows(A));
    return legacyVector<T, idx_t>(A.n, &A.ptr[rowIdx * A.ldim]);
}

// Get columns of legacyMatrix
template <typename T, class idx_t, Layout layout, class SliceSpec>
inline constexpr auto cols(const legacyMatrix<T, idx_t, layout>& A,
                           SliceSpec&& cols) noexcept
{
    assert((cols.first >= 0 and (idx_t) cols.first < ncols(A)) ||
           cols.first == cols.second);
    assert(cols.second >= 0 and (idx_t) cols.second <= ncols(A));
    assert(cols.first <= cols.second);
    return legacyMatrix<T, idx_t, layout>(A.m, cols.second - cols.first,
                                          (layout == Layout::ColMajor)
                                              ? &A.ptr[cols.first * A.ldim]
                                              : &A.ptr[cols.first],
                                          A.ldim);
}

// Get a column of a column-major legacyMatrix
template <typename T, class idx_t>
inline constexpr auto col(const legacyMatrix<T, idx_t>& A,
                          size_type<legacyMatrix<T, idx_t>> colIdx) noexcept
{
    assert(colIdx >= 0 and colIdx < ncols(A));
    return legacyVector<T, idx_t>(A.m, &A.ptr[colIdx * A.ldim]);
}

// Get a column of a row-major legacyMatrix
template <typename T, class idx_t>
inline constexpr auto col(
    const legacyMatrix<T, idx_t, Layout::RowMajor>& A,
    size_type<legacyMatrix<T, idx_t, Layout::RowMajor>> colIdx) noexcept
{
    assert(colIdx >= 0 and colIdx < ncols(A));
    return legacyVector<T, idx_t, idx_t>(A.m, &A.ptr[colIdx], A.ldim);
}

// Diagonal of a legacyMatrix
template <typename T, class idx_t, Layout layout>
inline constexpr auto diag(const legacyMatrix<T, idx_t, layout>& A,
                           int diagIdx = 0) noexcept
{
    assert(diagIdx >= 0 || (idx_t)(-diagIdx) < nrows(A));
    assert(diagIdx <= 0 || (idx_t)diagIdx < ncols(A));
    T* ptr = (diagIdx >= 0) ? &A(0, diagIdx) : &A(-diagIdx, 0);
    idx_t n = (diagIdx >= 0) ? std::min(A.m + diagIdx, A.n) - (idx_t)diagIdx
                             : std::min(A.m, A.n - diagIdx) + (idx_t)diagIdx;

    return legacyVector<T, idx_t, idx_t>(n, ptr, A.ldim + 1);
}

// slice legacyVector
template <typename T,
          class idx_t,
          typename int_t,
          Direction direction,
          class SliceSpec>
inline constexpr auto slice(const legacyVector<T, idx_t, int_t, direction>& v,
                            SliceSpec&& rows) noexcept
{
    assert((rows.first >= 0 and (idx_t) rows.first < size(v)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 and (idx_t) rows.second <= size(v));
    assert(rows.first <= rows.second);
    return legacyVector<T, idx_t, int_t, direction>(
        rows.second - rows.first, &v.ptr[rows.first * v.inc], v.inc);
}

// -----------------------------------------------------------------------------
// Deduce matrix and vector type from two provided ones

namespace internal {

#ifdef TLAPACK_PREFERRED_MATRIX_LEGACY

    #ifndef TLAPACK_EIGEN_HH
        #ifndef TLAPACK_MDSPAN_HH
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) true
        #else
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) !is_mdspan_type<T>
        #endif
    #else
        #ifndef TLAPACK_MDSPAN_HH
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) !is_eigen_type<T>
        #else
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                (!is_eigen_type<T> && !is_mdspan_type<T>)
        #endif
    #endif

    // for two types
    // should be especialized for every new matrix class
    template <class matrixA_t, typename matrixB_t>
    struct matrix_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<TLAPACK_USE_PREFERRED_MATRIX_TYPE(matrixA_t) ||
                                    TLAPACK_USE_PREFERRED_MATRIX_TYPE(
                                        matrixB_t),
                                int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;

        static constexpr Layout LA = layout<matrixA_t>;
        static constexpr Layout LB = layout<matrixB_t>;
        static constexpr Layout L =
            ((LA == Layout::RowMajor) && (LB == Layout::RowMajor))
                ? Layout::RowMajor
                : Layout::ColMajor;

        using type = legacyMatrix<T, idx_t, L>;
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
        using idx_t = size_type<vecA_t>;

        using type = legacyVector<T, idx_t, idx_t>;
    };

    #undef TLAPACK_USE_PREFERRED_MATRIX_TYPE

#else

    // for two types
    // should be especialized for every new matrix class
    template <class matrixA_t, class matrixB_t>
    struct matrix_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<is_legacy_type<matrixA_t> &&
                                    is_legacy_type<matrixB_t>,
                                int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;

        static constexpr Layout LA = layout<matrixA_t>;
        static constexpr Layout LB = layout<matrixB_t>;
        static constexpr Layout L =
            ((LA == Layout::RowMajor) && (LB == Layout::RowMajor))
                ? Layout::RowMajor
                : Layout::ColMajor;

        using type = legacyMatrix<T, idx_t, L>;
    };

    // for two types
    // should be especialized for every new vector class
    template <class matrixA_t, class matrixB_t>
    struct vector_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<is_legacy_type<matrixA_t> &&
                                    is_legacy_type<matrixB_t>,
                                int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;

        using type = legacyVector<T, idx_t, idx_t>;
    };

#endif  // TLAPACK_PREFERRED_MATRIX
}  // namespace internal

// -----------------------------------------------------------------------------
// Cast to Legacy arrays

template <typename T, class idx_t, Layout layout>
inline constexpr auto legacy_matrix(
    const legacyMatrix<T, idx_t, layout>& A) noexcept
{
    return legacy::matrix<T, idx_t>{layout, A.m, A.n, A.ptr, A.ldim};
}

template <class T, class idx_t, class int_t, Direction direction>
inline constexpr auto legacy_matrix(
    const legacyVector<T, idx_t, int_t, direction>& v) noexcept
{
    return legacy::matrix<T, idx_t>{Layout::ColMajor, 1, v.n, v.ptr, v.inc};
}

template <typename T, class idx_t, typename int_t, Direction direction>
inline constexpr auto legacy_vector(
    const legacyVector<T, idx_t, int_t, direction>& v) noexcept
{
    assert(direction == Direction::Forward || std::is_signed<idx_t>::value ||
           v.inc == 0);

    return legacy::vector<T, idx_t>{
        v.n, v.ptr,
        (direction == Direction::Forward) ? idx_t(v.inc) : idx_t(-v.inc)};
}

}  // namespace tlapack

#endif  // TLAPACK_LEGACYARRAY_HH
