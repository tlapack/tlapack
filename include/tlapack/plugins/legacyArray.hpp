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

#include "tlapack/LegacyBandedMatrix.hpp"
#include "tlapack/LegacyMatrix.hpp"
#include "tlapack/LegacyVector.hpp"
#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/workspace.hpp"

namespace tlapack {

// -----------------------------------------------------------------------------
// Helpers

namespace legacy {
    namespace internal {
        template <typename T, class idx_t, Layout L>
        std::true_type is_legacy_type_f(const LegacyMatrix<T, idx_t, L>*);

        template <typename T, class idx_t, class int_t, Direction D>
        std::true_type is_legacy_type_f(
            const LegacyVector<T, idx_t, int_t, D>*);

        std::false_type is_legacy_type_f(const void*);
    }  // namespace internal

    /// True if T is a legacy array
    /// @see https://stackoverflow.com/a/25223400/5253097
    template <class T>
    constexpr bool is_legacy_type =
        decltype(internal::is_legacy_type_f(std::declval<T*>()))::value;
}  // namespace legacy

// -----------------------------------------------------------------------------
// Data traits

namespace traits {
    /// Layout for LegacyMatrix
    template <typename T, class idx_t, Layout L>
    struct layout_trait<LegacyMatrix<T, idx_t, L>, int> {
        static constexpr Layout value = L;
    };

    /// Layout for LegacyVector
    template <typename T, class idx_t, typename int_t, Direction D>
    struct layout_trait<LegacyVector<T, idx_t, int_t, D>, int> {
        static constexpr Layout value = Layout::Strided;
    };

    template <class T, class idx_t, typename int_t, Direction D>
    struct real_type_traits<LegacyVector<T, idx_t, int_t, D>, int> {
        using type = LegacyVector<real_type<T>, idx_t, int_t, D>;
    };

    template <typename T, class idx_t, Layout layout>
    struct real_type_traits<LegacyMatrix<T, idx_t, layout>, int> {
        using type = LegacyMatrix<real_type<T>, idx_t, layout>;
    };

    template <class T, class idx_t, typename int_t, Direction D>
    struct complex_type_traits<LegacyVector<T, idx_t, int_t, D>, int> {
        using type = LegacyVector<complex_type<T>, idx_t, int_t, D>;
    };

    template <typename T, class idx_t, Layout layout>
    struct complex_type_traits<LegacyMatrix<T, idx_t, layout>, int> {
        using type = LegacyMatrix<complex_type<T>, idx_t, layout>;
    };

    /// Create LegacyMatrix @see Create
    template <class T, class idx_t, Layout layout>
    struct CreateFunctor<LegacyMatrix<T, idx_t, layout>, int> {
        inline constexpr auto operator()(std::vector<T>& v,
                                         idx_t m,
                                         idx_t n) const
        {
            using matrix_t = LegacyMatrix<T, idx_t, layout>;

            assert(m >= 0 && n >= 0);
            v.resize(m * n);  // Allocates space in memory
            return matrix_t(m, n, v.data());
        }

        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         idx_t n,
                                         Workspace& rW) const
        {
            using matrix_t = LegacyMatrix<T, idx_t, Layout::ColMajor>;

            assert(m >= 0 && n >= 0);
            rW = W.extract(m * sizeof(T), n);
            return (W.isContiguous())
                       ? matrix_t(m, n, (T*)W.data())
                       : matrix_t(m, n, (T*)W.data(), W.getLdim() / sizeof(T));
        }

        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         idx_t n) const
        {
            using matrix_t = LegacyMatrix<T, idx_t, Layout::ColMajor>;

            assert(m >= 0 && n >= 0);
            tlapack_check(W.contains(m * sizeof(T), n));
            return (W.isContiguous())
                       ? matrix_t(m, n, (T*)W.data())
                       : matrix_t(m, n, (T*)W.data(), W.getLdim() / sizeof(T));
        }
    };

    /// Create LegacyVector @see Create
    template <class T, class idx_t, typename int_t, Direction D>
    struct CreateFunctor<LegacyVector<T, idx_t, int_t, D>, int> {
        using vector_t = LegacyVector<T, idx_t, int_t, D>;

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
                       ? vector_t(m, (T*)W.data())
                       : vector_t(m, (T*)W.data(), W.getLdim() / sizeof(T));
        }

        inline constexpr auto operator()(const Workspace& W, idx_t m) const
        {
            assert(m >= 0);
            tlapack_check(W.contains(sizeof(T), m));
            return (W.isContiguous())
                       ? vector_t(m, (T*)W.data())
                       : vector_t(m, (T*)W.data(), W.getLdim() / sizeof(T));
        }
    };
}  // namespace traits

// -----------------------------------------------------------------------------
// Data descriptors

// Number of rows of LegacyMatrix
template <typename T, class idx_t, Layout layout>
inline constexpr auto nrows(const LegacyMatrix<T, idx_t, layout>& A)
{
    return A.m;
}

// Number of columns of LegacyMatrix
template <typename T, class idx_t, Layout layout>
inline constexpr auto ncols(const LegacyMatrix<T, idx_t, layout>& A)
{
    return A.n;
}

// Size of LegacyVector
template <typename T, class idx_t, typename int_t, Direction direction>
inline constexpr auto size(const LegacyVector<T, idx_t, int_t, direction>& x)
{
    return x.n;
}

// Number of rows of LegacyBandedMatrix
template <typename T, class idx_t>
inline constexpr auto nrows(const LegacyBandedMatrix<T, idx_t>& A)
{
    return A.m;
}

// Number of columns of LegacyBandedMatrix
template <typename T, class idx_t>
inline constexpr auto ncols(const LegacyBandedMatrix<T, idx_t>& A)
{
    return A.n;
}

// Lowerband of LegacyBandedMatrix
template <typename T, class idx_t>
inline constexpr auto lowerband(const LegacyBandedMatrix<T, idx_t>& A)
{
    return A.kl;
}

// Upperband of LegacyBandedMatrix
template <typename T, class idx_t>
inline constexpr auto upperband(const LegacyBandedMatrix<T, idx_t>& A)
{
    return A.ku;
}

// -----------------------------------------------------------------------------
// Block operations for const LegacyMatrix

#define isSlice(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value

// Slice LegacyMatrix
template <
    typename T,
    class idx_t,
    Layout layout,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
                            int>::type = 0>
inline constexpr auto slice(const LegacyMatrix<T, idx_t, layout>& A,
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
    return LegacyMatrix<const T, idx_t, layout>(
        rows.second - rows.first, cols.second - cols.first,
        (layout == Layout::ColMajor) ? &A.ptr[rows.first + cols.first * A.ldim]
                                     : &A.ptr[rows.first * A.ldim + cols.first],
        A.ldim);
}

#undef isSlice

// Slice LegacyMatrix over a single row
template <typename T, class idx_t, Layout layout, class SliceSpecCol>
inline constexpr auto slice(const LegacyMatrix<T, idx_t, layout>& A,
                            size_type<LegacyMatrix<T, idx_t, layout>> rowIdx,
                            SliceSpecCol&& cols) noexcept
{
    assert((cols.first >= 0 and (idx_t) cols.first < ncols(A)) ||
           cols.first == cols.second);
    assert(cols.second >= 0 and (idx_t) cols.second <= ncols(A));
    assert(cols.first <= cols.second);
    assert(rowIdx >= 0 and rowIdx < nrows(A));
    return LegacyVector<const T, idx_t, idx_t>(
        cols.second - cols.first,
        (layout == Layout::ColMajor) ? &A.ptr[rowIdx + cols.first * A.ldim]
                                     : &A.ptr[rowIdx * A.ldim + cols.first],
        layout == Layout::ColMajor ? A.ldim : 1);
}

// Slice LegacyMatrix over a single column
template <typename T, class idx_t, Layout layout, class SliceSpecRow>
inline constexpr auto slice(
    const LegacyMatrix<T, idx_t, layout>& A,
    SliceSpecRow&& rows,
    size_type<LegacyMatrix<T, idx_t, layout>> colIdx) noexcept
{
    assert((rows.first >= 0 and (idx_t) rows.first < nrows(A)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 and (idx_t) rows.second <= nrows(A));
    assert(rows.first <= rows.second);
    assert(colIdx >= 0 and colIdx < ncols(A));
    return LegacyVector<const T, idx_t, idx_t>(
        rows.second - rows.first,
        (layout == Layout::ColMajor) ? &A.ptr[rows.first + colIdx * A.ldim]
                                     : &A.ptr[rows.first * A.ldim + colIdx],
        layout == Layout::RowMajor ? A.ldim : 1);
}

// Get rows of LegacyMatrix
template <typename T, class idx_t, Layout layout, class SliceSpec>
inline constexpr auto rows(const LegacyMatrix<T, idx_t, layout>& A,
                           SliceSpec&& rows) noexcept
{
    assert((rows.first >= 0 and (idx_t) rows.first < nrows(A)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 and (idx_t) rows.second <= nrows(A));
    assert(rows.first <= rows.second);
    return LegacyMatrix<const T, idx_t, layout>(
        rows.second - rows.first, A.n,
        (layout == Layout::ColMajor) ? &A.ptr[rows.first]
                                     : &A.ptr[rows.first * A.ldim],
        A.ldim);
}

// Get a row of a column-major LegacyMatrix
template <typename T, class idx_t>
inline constexpr auto row(const LegacyMatrix<T, idx_t>& A,
                          size_type<LegacyMatrix<T, idx_t>> rowIdx) noexcept
{
    assert(rowIdx >= 0 and rowIdx < nrows(A));
    return LegacyVector<const T, idx_t, idx_t>(A.n, &A.ptr[rowIdx], A.ldim);
}

// Get a row of a row-major LegacyMatrix
template <typename T, class idx_t>
inline constexpr auto row(
    const LegacyMatrix<T, idx_t, Layout::RowMajor>& A,
    size_type<LegacyMatrix<T, idx_t, Layout::RowMajor>> rowIdx) noexcept
{
    assert(rowIdx >= 0 and rowIdx < nrows(A));
    return LegacyVector<const T, idx_t>(A.n, &A.ptr[rowIdx * A.ldim]);
}

// Get columns of LegacyMatrix
template <typename T, class idx_t, Layout layout, class SliceSpec>
inline constexpr auto cols(const LegacyMatrix<T, idx_t, layout>& A,
                           SliceSpec&& cols) noexcept
{
    assert((cols.first >= 0 and (idx_t) cols.first < ncols(A)) ||
           cols.first == cols.second);
    assert(cols.second >= 0 and (idx_t) cols.second <= ncols(A));
    assert(cols.first <= cols.second);
    return LegacyMatrix<const T, idx_t, layout>(
        A.m, cols.second - cols.first,
        (layout == Layout::ColMajor) ? &A.ptr[cols.first * A.ldim]
                                     : &A.ptr[cols.first],
        A.ldim);
}

// Get a column of a column-major LegacyMatrix
template <typename T, class idx_t>
inline constexpr auto col(const LegacyMatrix<T, idx_t>& A,
                          size_type<LegacyMatrix<T, idx_t>> colIdx) noexcept
{
    assert(colIdx >= 0 and colIdx < ncols(A));
    return LegacyVector<const T, idx_t>(A.m, &A.ptr[colIdx * A.ldim]);
}

// Get a column of a row-major LegacyMatrix
template <typename T, class idx_t>
inline constexpr auto col(
    const LegacyMatrix<T, idx_t, Layout::RowMajor>& A,
    size_type<LegacyMatrix<T, idx_t, Layout::RowMajor>> colIdx) noexcept
{
    assert(colIdx >= 0 and colIdx < ncols(A));
    return LegacyVector<const T, idx_t, idx_t>(A.m, &A.ptr[colIdx], A.ldim);
}

// Diagonal of a LegacyMatrix
template <typename T, class idx_t, Layout layout>
inline constexpr auto diag(const LegacyMatrix<T, idx_t, layout>& A,
                           int diagIdx = 0) noexcept
{
    assert(diagIdx >= 0 || (idx_t)(-diagIdx) < nrows(A));
    assert(diagIdx <= 0 || (idx_t)diagIdx < ncols(A));
    T* ptr =
        (layout == Layout::ColMajor)
            ? ((diagIdx >= 0) ? &A.ptr[diagIdx * A.ldim] : &A.ptr[-diagIdx])
            : ((diagIdx >= 0) ? &A.ptr[diagIdx] : &A.ptr[-diagIdx * A.ldim]);
    idx_t n = (diagIdx >= 0) ? std::min(A.m + diagIdx, A.n) - (idx_t)diagIdx
                             : std::min(A.m, A.n - diagIdx) + (idx_t)diagIdx;

    return LegacyVector<const T, idx_t, idx_t>(n, ptr, A.ldim + 1);
}

// Transpose view of a LegacyMatrix
template <typename T, class idx_t>
inline constexpr auto transpose_view(
    const LegacyMatrix<T, idx_t, Layout::ColMajor>& A) noexcept
{
    return LegacyMatrix<const T, idx_t, Layout::RowMajor>(A.n, A.m, A.ptr,
                                                          A.ldim);
}
template <typename T, class idx_t>
inline constexpr auto transpose_view(
    const LegacyMatrix<T, idx_t, Layout::RowMajor>& A) noexcept
{
    return LegacyMatrix<const T, idx_t, Layout::ColMajor>(A.n, A.m, A.ptr,
                                                          A.ldim);
}

// slice LegacyVector
template <typename T,
          class idx_t,
          typename int_t,
          Direction direction,
          class SliceSpec>
inline constexpr auto slice(const LegacyVector<T, idx_t, int_t, direction>& v,
                            SliceSpec&& rows) noexcept
{
    assert((rows.first >= 0 and (idx_t) rows.first < size(v)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 and (idx_t) rows.second <= size(v));
    assert(rows.first <= rows.second);
    return LegacyVector<const T, idx_t, int_t, direction>(
        rows.second - rows.first, &v.ptr[rows.first * v.inc], v.inc);
}

// -----------------------------------------------------------------------------
// Block operations for non-const LegacyMatrix

#define isSlice(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value

// Slice LegacyMatrix
template <
    typename T,
    class idx_t,
    Layout layout,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
                            int>::type = 0>
inline constexpr auto slice(LegacyMatrix<T, idx_t, layout>& A,
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
    return LegacyMatrix<T, idx_t, layout>(
        rows.second - rows.first, cols.second - cols.first,
        (layout == Layout::ColMajor) ? &A.ptr[rows.first + cols.first * A.ldim]
                                     : &A.ptr[rows.first * A.ldim + cols.first],
        A.ldim);
}

#undef isSlice

// Slice LegacyMatrix over a single row
template <typename T, class idx_t, Layout layout, class SliceSpecCol>
inline constexpr auto slice(LegacyMatrix<T, idx_t, layout>& A,
                            size_type<LegacyMatrix<T, idx_t, layout>> rowIdx,
                            SliceSpecCol&& cols) noexcept
{
    assert((cols.first >= 0 and (idx_t) cols.first < ncols(A)) ||
           cols.first == cols.second);
    assert(cols.second >= 0 and (idx_t) cols.second <= ncols(A));
    assert(cols.first <= cols.second);
    assert(rowIdx >= 0 and rowIdx < nrows(A));
    return LegacyVector<T, idx_t, idx_t>(
        cols.second - cols.first,
        (layout == Layout::ColMajor) ? &A.ptr[rowIdx + cols.first * A.ldim]
                                     : &A.ptr[rowIdx * A.ldim + cols.first],
        layout == Layout::ColMajor ? A.ldim : 1);
}

// Slice LegacyMatrix over a single column
template <typename T, class idx_t, Layout layout, class SliceSpecRow>
inline constexpr auto slice(
    LegacyMatrix<T, idx_t, layout>& A,
    SliceSpecRow&& rows,
    size_type<LegacyMatrix<T, idx_t, layout>> colIdx) noexcept
{
    assert((rows.first >= 0 and (idx_t) rows.first < nrows(A)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 and (idx_t) rows.second <= nrows(A));
    assert(rows.first <= rows.second);
    assert(colIdx >= 0 and colIdx < ncols(A));
    return LegacyVector<T, idx_t, idx_t>(
        rows.second - rows.first,
        (layout == Layout::ColMajor) ? &A.ptr[rows.first + colIdx * A.ldim]
                                     : &A.ptr[rows.first * A.ldim + colIdx],
        layout == Layout::RowMajor ? A.ldim : 1);
}

// Get rows of LegacyMatrix
template <typename T, class idx_t, Layout layout, class SliceSpec>
inline constexpr auto rows(LegacyMatrix<T, idx_t, layout>& A,
                           SliceSpec&& rows) noexcept
{
    assert((rows.first >= 0 and (idx_t) rows.first < nrows(A)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 and (idx_t) rows.second <= nrows(A));
    assert(rows.first <= rows.second);
    return LegacyMatrix<T, idx_t, layout>(rows.second - rows.first, A.n,
                                          (layout == Layout::ColMajor)
                                              ? &A.ptr[rows.first]
                                              : &A.ptr[rows.first * A.ldim],
                                          A.ldim);
}

// Get a row of a column-major LegacyMatrix
template <typename T, class idx_t>
inline constexpr auto row(LegacyMatrix<T, idx_t>& A,
                          size_type<LegacyMatrix<T, idx_t>> rowIdx) noexcept
{
    assert(rowIdx >= 0 and rowIdx < nrows(A));
    return LegacyVector<T, idx_t, idx_t>(A.n, &A.ptr[rowIdx], A.ldim);
}

// Get a row of a row-major LegacyMatrix
template <typename T, class idx_t>
inline constexpr auto row(
    LegacyMatrix<T, idx_t, Layout::RowMajor>& A,
    size_type<LegacyMatrix<T, idx_t, Layout::RowMajor>> rowIdx) noexcept
{
    assert(rowIdx >= 0 and rowIdx < nrows(A));
    return LegacyVector<T, idx_t>(A.n, &A.ptr[rowIdx * A.ldim]);
}

// Get columns of LegacyMatrix
template <typename T, class idx_t, Layout layout, class SliceSpec>
inline constexpr auto cols(LegacyMatrix<T, idx_t, layout>& A,
                           SliceSpec&& cols) noexcept
{
    assert((cols.first >= 0 and (idx_t) cols.first < ncols(A)) ||
           cols.first == cols.second);
    assert(cols.second >= 0 and (idx_t) cols.second <= ncols(A));
    assert(cols.first <= cols.second);
    return LegacyMatrix<T, idx_t, layout>(A.m, cols.second - cols.first,
                                          (layout == Layout::ColMajor)
                                              ? &A.ptr[cols.first * A.ldim]
                                              : &A.ptr[cols.first],
                                          A.ldim);
}

// Get a column of a column-major LegacyMatrix
template <typename T, class idx_t>
inline constexpr auto col(LegacyMatrix<T, idx_t>& A,
                          size_type<LegacyMatrix<T, idx_t>> colIdx) noexcept
{
    assert(colIdx >= 0 and colIdx < ncols(A));
    return LegacyVector<T, idx_t>(A.m, &A.ptr[colIdx * A.ldim]);
}

// Get a column of a row-major LegacyMatrix
template <typename T, class idx_t>
inline constexpr auto col(
    LegacyMatrix<T, idx_t, Layout::RowMajor>& A,
    size_type<LegacyMatrix<T, idx_t, Layout::RowMajor>> colIdx) noexcept
{
    assert(colIdx >= 0 and colIdx < ncols(A));
    return LegacyVector<T, idx_t, idx_t>(A.m, &A.ptr[colIdx], A.ldim);
}

// Diagonal of a LegacyMatrix
template <typename T, class idx_t, Layout layout>
inline constexpr auto diag(LegacyMatrix<T, idx_t, layout>& A,
                           int diagIdx = 0) noexcept
{
    assert(diagIdx >= 0 || (idx_t)(-diagIdx) < nrows(A));
    assert(diagIdx <= 0 || (idx_t)diagIdx < ncols(A));
    T* ptr =
        (layout == Layout::ColMajor)
            ? ((diagIdx >= 0) ? &A.ptr[diagIdx * A.ldim] : &A.ptr[-diagIdx])
            : ((diagIdx >= 0) ? &A.ptr[diagIdx] : &A.ptr[-diagIdx * A.ldim]);
    idx_t n = (diagIdx >= 0) ? std::min(A.m + diagIdx, A.n) - (idx_t)diagIdx
                             : std::min(A.m, A.n - diagIdx) + (idx_t)diagIdx;

    return LegacyVector<T, idx_t, idx_t>(n, ptr, A.ldim + 1);
}

// Transpose view of a LegacyMatrix
template <typename T, class idx_t>
inline constexpr auto transpose_view(
    LegacyMatrix<T, idx_t, Layout::ColMajor>& A) noexcept
{
    return LegacyMatrix<T, idx_t, Layout::RowMajor>(A.n, A.m, A.ptr, A.ldim);
}
template <typename T, class idx_t>
inline constexpr auto transpose_view(
    LegacyMatrix<T, idx_t, Layout::RowMajor>& A) noexcept
{
    return LegacyMatrix<T, idx_t, Layout::ColMajor>(A.n, A.m, A.ptr, A.ldim);
}

// slice LegacyVector
template <typename T,
          class idx_t,
          typename int_t,
          Direction direction,
          class SliceSpec>
inline constexpr auto slice(LegacyVector<T, idx_t, int_t, direction>& v,
                            SliceSpec&& rows) noexcept
{
    assert((rows.first >= 0 and (idx_t) rows.first < size(v)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 and (idx_t) rows.second <= size(v));
    assert(rows.first <= rows.second);
    return LegacyVector<T, idx_t, int_t, direction>(
        rows.second - rows.first, &v.ptr[rows.first * v.inc], v.inc);
}

// -----------------------------------------------------------------------------
// Deduce matrix and vector type from two provided ones

namespace traits {

#ifdef TLAPACK_PREFERRED_MATRIX_LEGACY

    #ifndef TLAPACK_EIGEN_HH
        #ifndef TLAPACK_MDSPAN_HH
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) true
        #else
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                !mdspan::is_mdspan_type<T>
        #endif
    #else
        #ifndef TLAPACK_MDSPAN_HH
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                !eigen::is_eigen_type<T>
        #else
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                (!eigen::is_eigen_type<T> && !mdspan::is_mdspan_type<T>)
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

        using type = LegacyMatrix<T, idx_t, L>;
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

        using type = LegacyVector<T, idx_t, idx_t>;
    };

    #undef TLAPACK_USE_PREFERRED_MATRIX_TYPE

#else

    // for two types
    // should be especialized for every new matrix class
    template <class matrixA_t, class matrixB_t>
    struct matrix_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<legacy::is_legacy_type<matrixA_t> &&
                                    legacy::is_legacy_type<matrixB_t>,
                                int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;

        static constexpr Layout LA = layout<matrixA_t>;
        static constexpr Layout LB = layout<matrixB_t>;
        static constexpr Layout L =
            ((LA == Layout::RowMajor) && (LB == Layout::RowMajor))
                ? Layout::RowMajor
                : Layout::ColMajor;

        using type = LegacyMatrix<T, idx_t, L>;
    };

    // for two types
    // should be especialized for every new vector class
    template <class matrixA_t, class matrixB_t>
    struct vector_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<legacy::is_legacy_type<matrixA_t> &&
                                    legacy::is_legacy_type<matrixB_t>,
                                int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;

        using type = LegacyVector<T, idx_t, idx_t>;
    };

#endif  // TLAPACK_PREFERRED_MATRIX
}  // namespace traits

// -----------------------------------------------------------------------------
// Cast to Legacy arrays

template <typename T, class idx_t, Layout layout>
inline constexpr auto legacy_matrix(
    const LegacyMatrix<T, idx_t, layout>& A) noexcept
{
    return legacy::Matrix<T, idx_t>{layout, A.m, A.n, A.ptr, A.ldim};
}

template <class T, class idx_t, class int_t, Direction direction>
inline constexpr auto legacy_matrix(
    const LegacyVector<T, idx_t, int_t, direction>& v) noexcept
{
    return legacy::Matrix<T, idx_t>{Layout::ColMajor, 1, v.n, v.ptr,
                                    (idx_t)v.inc};
}

template <typename T, class idx_t, typename int_t, Direction direction>
inline constexpr auto legacy_vector(
    const LegacyVector<T, idx_t, int_t, direction>& v) noexcept
{
    assert(direction == Direction::Forward || std::is_signed<idx_t>::value ||
           v.inc == 0);

    return legacy::Vector<T, idx_t>{
        v.n, v.ptr,
        (direction == Direction::Forward) ? idx_t(v.inc) : idx_t(-v.inc)};
}

}  // namespace tlapack

#endif  // TLAPACK_LEGACYARRAY_HH
