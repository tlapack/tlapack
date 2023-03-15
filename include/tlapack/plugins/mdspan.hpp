/// @file mdspan.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MDSPAN_HH
#define TLAPACK_MDSPAN_HH

#include <cassert>
#include <experimental/mdspan>

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/workspace.hpp"

namespace tlapack {

// -----------------------------------------------------------------------------
// Helpers

namespace internal {
    template <class ET, class Exts, class LP, class AP>
    std::true_type is_mdspan_type_f(
        const std::experimental::mdspan<ET, Exts, LP, AP>*);

    std::false_type is_mdspan_type_f(const void*);
}  // namespace internal

/// True if T is a mdspan array
/// @see https://stackoverflow.com/a/25223400/5253097
template <class T>
constexpr bool is_mdspan_type =
    decltype(internal::is_mdspan_type_f(std::declval<T*>()))::value;

// -----------------------------------------------------------------------------
// Data traits

namespace internal {
    /// Layout for mdspan
    template <class ET, class Exts, class AP>
    struct LayoutImpl<
        std::experimental::mdspan<ET, Exts, std::experimental::layout_left, AP>,
        std::enable_if_t<Exts::rank() == 2, int>> {
        static constexpr Layout layout = Layout::ColMajor;
    };
    template <class ET, class Exts, class AP>
    struct LayoutImpl<std::experimental::
                          mdspan<ET, Exts, std::experimental::layout_right, AP>,
                      std::enable_if_t<Exts::rank() == 2, int>> {
        static constexpr Layout layout = Layout::RowMajor;
    };
    template <class ET, class Exts, class LP, class AP>
    struct LayoutImpl<
        std::experimental::mdspan<ET, Exts, LP, AP>,
        std::enable_if_t<(Exts::rank() == 1) &&
                             LP::template mapping<Exts>::is_always_strided(),
                         int>> {
        static constexpr Layout layout = Layout::Strided;
    };

    /// Transpose type for legacyMatrix
    template <class ET, class Exts, class AP>
    struct TransposeTypeImpl<
        std::experimental::mdspan<ET, Exts, std::experimental::layout_left, AP>,
        int> {
        using type = std::experimental::
            mdspan<ET, Exts, std::experimental::layout_right, AP>;
    };
    template <class ET, class Exts, class AP>
    struct TransposeTypeImpl<
        std::experimental::
            mdspan<ET, Exts, std::experimental::layout_right, AP>,
        int> {
        using type = std::experimental::
            mdspan<ET, Exts, std::experimental::layout_left, AP>;
    };
    template <class ET, class Exts, class AP>
    struct TransposeTypeImpl<
        std::experimental::
            mdspan<ET, Exts, std::experimental::layout_stride, AP>,
        int> {
        using type = std::experimental::
            mdspan<ET, Exts, std::experimental::layout_stride, AP>;
    };

    /// Create mdspan @see Create
    template <class ET, class Exts, class LP, class AP>
    struct CreateImpl<std::experimental::mdspan<ET, Exts, LP, AP>,
                      std::enable_if_t<Exts::rank() == 1, int>> {
        using idx_t =
            typename std::experimental::mdspan<ET, Exts, LP>::size_type;
        using extents_t = std::experimental::dextents<idx_t, 1>;

        inline constexpr auto operator()(std::vector<ET>& v, idx_t m) const
        {
            assert(m >= 0);
            v.resize(m);  // Allocates space in memory
            return std::experimental::mdspan<ET, extents_t>(v.data(), m);
        }

        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         Workspace& rW) const
        {
            using std::array;
            using std::experimental::layout_stride;
            using mapping = typename layout_stride::template mapping<extents_t>;
            using matrix_t =
                std::experimental::mdspan<ET, extents_t, layout_stride>;

            assert(m >= 0);

            rW = W.extract(sizeof(ET), m);
            mapping map =
                (W.isContiguous())
                    ? mapping(extents_t(m), array<idx_t, 1>{1})
                    : mapping(extents_t(m),
                              array<idx_t, 1>{W.getLdim() / sizeof(ET)});

            return matrix_t((ET*)W.data(), std::move(map));
        }

        inline constexpr auto operator()(const Workspace& W, idx_t m) const
        {
            using std::array;
            using std::experimental::layout_stride;
            using mapping = typename layout_stride::template mapping<extents_t>;
            using matrix_t =
                std::experimental::mdspan<ET, extents_t, layout_stride>;

            assert(m >= 0);

            tlapack_check(W.contains(sizeof(ET), m));
            mapping map =
                (W.isContiguous())
                    ? mapping(extents_t(m), array<idx_t, 1>{1})
                    : mapping(extents_t(m),
                              array<idx_t, 1>{W.getLdim() / sizeof(ET)});

            return matrix_t((ET*)W.data(), std::move(map));
        }
    };

    /// Create mdspan @see Create
    template <class ET, class Exts, class LP, class AP>
    struct CreateImpl<std::experimental::mdspan<ET, Exts, LP, AP>,
                      std::enable_if_t<Exts::rank() == 2, int>> {
        using idx_t =
            typename std::experimental::mdspan<ET, Exts, LP>::size_type;
        using extents_t = std::experimental::dextents<idx_t, 2>;

        inline constexpr auto operator()(std::vector<ET>& v,
                                         idx_t m,
                                         idx_t n) const
        {
            assert(m >= 0 && n >= 0);
            v.resize(m * n);  // Allocates space in memory
            return std::experimental::mdspan<ET, extents_t>(v.data(), m, n);
        }

        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         idx_t n,
                                         Workspace& rW) const
        {
            using std::array;
            using std::experimental::layout_stride;
            using mapping = typename layout_stride::template mapping<extents_t>;
            using matrix_t =
                std::experimental::mdspan<ET, extents_t, layout_stride>;

            assert(m >= 0 && n >= 0);

            // Variables to be forwarded to the returned matrix
            array<idx_t, 2> strides = [&](Workspace& rW) {
                if (W.isContiguous()) {
                    rW = W.extract(m * sizeof(ET), n);
                    return array<idx_t, 2>{1, m};
                }
                else if (W.getM() >= m * sizeof(ET) && W.getN() >= n) {
                    rW = W.extract(m * sizeof(ET), n);
                    return array<idx_t, 2>{1, W.getLdim() / sizeof(ET)};
                }
                else {
                    rW = W.extract(n * sizeof(ET), m);
                    return array<idx_t, 2>{W.getLdim() / sizeof(ET), 1};
                }
            }(rW);
            mapping map = mapping(extents_t(m, n), std::move(strides));

            return matrix_t((ET*)W.data(), std::move(map));
        }

        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         idx_t n) const
        {
            using std::array;
            using std::experimental::layout_stride;
            using mapping = typename layout_stride::template mapping<extents_t>;
            using matrix_t =
                std::experimental::mdspan<ET, extents_t, layout_stride, AP>;

            assert(m >= 0 && n >= 0);

            // Variables to be forwarded to the returned matrix
            array<idx_t, 2> strides = [&]() {
                if (W.isContiguous()) {
                    tlapack_check(W.contains(m * sizeof(ET), n));
                    return array<idx_t, 2>{1, m};
                }
                else if (W.getM() >= m * sizeof(ET) && W.getN() >= n) {
                    tlapack_check(W.contains(m * sizeof(ET), n));
                    return array<idx_t, 2>{1, W.getLdim() / sizeof(ET)};
                }
                else {
                    tlapack_check(W.contains(n * sizeof(ET), m));
                    return array<idx_t, 2>{W.getLdim() / sizeof(ET), 1};
                }
            }();
            mapping map = mapping(extents_t(m, n), std::move(strides));

            return matrix_t((ET*)W.data(), std::move(map));
        }
    };
}  // namespace internal

// -----------------------------------------------------------------------------
// Data descriptors

// Size
template <class ET, class Exts, class LP, class AP>
inline constexpr auto size(const std::experimental::mdspan<ET, Exts, LP, AP>& x)
{
    return x.size();
}
// Number of rows
template <class ET, class Exts, class LP, class AP>
inline constexpr auto nrows(
    const std::experimental::mdspan<ET, Exts, LP, AP>& x)
{
    return x.extent(0);
}
// Number of columns
template <class ET, class Exts, class LP, class AP>
inline constexpr auto ncols(
    const std::experimental::mdspan<ET, Exts, LP, AP>& x)
{
    return x.extent(1);
}

// -----------------------------------------------------------------------------
// Block operations

#define isSlice(SliceSpec) \
    std::is_convertible<SliceSpec, std::tuple<std::size_t, std::size_t>>::value

// Slice
template <
    class ET,
    class Exts,
    class LP,
    class AP,
    class SliceSpecRow,
    class SliceSpecCol,
    std::enable_if_t<isSlice(SliceSpecRow) || isSlice(SliceSpecCol), int> = 0>
inline constexpr auto slice(
    const std::experimental::mdspan<ET, Exts, LP, AP>& A,
    SliceSpecRow&& rows,
    SliceSpecCol&& cols) noexcept
{
    return std::experimental::submdspan(A, std::forward<SliceSpecRow>(rows),
                                        std::forward<SliceSpecCol>(cols));
}

// Rows
template <class ET,
          class Exts,
          class LP,
          class AP,
          class SliceSpec,
          std::enable_if_t<isSlice(SliceSpec), int> = 0>
inline constexpr auto rows(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
                           SliceSpec&& rows) noexcept
{
    return std::experimental::submdspan(A, std::forward<SliceSpec>(rows),
                                        std::experimental::full_extent);
}

// Row
template <class ET, class Exts, class LP, class AP>
inline constexpr auto row(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
                          std::size_t rowIdx) noexcept
{
    return std::experimental::submdspan(A, rowIdx,
                                        std::experimental::full_extent);
}

// Columns
template <class ET,
          class Exts,
          class LP,
          class AP,
          class SliceSpec,
          std::enable_if_t<isSlice(SliceSpec), int> = 0>
inline constexpr auto cols(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
                           SliceSpec&& cols) noexcept
{
    return std::experimental::submdspan(A, std::experimental::full_extent,
                                        std::forward<SliceSpec>(cols));
}

// Column
template <class ET, class Exts, class LP, class AP>
inline constexpr auto col(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
                          std::size_t colIdx) noexcept
{
    return std::experimental::submdspan(A, std::experimental::full_extent,
                                        colIdx);
}

// Slice
template <class ET,
          class Exts,
          class LP,
          class AP,
          class SliceSpec,
          std::enable_if_t<isSlice(SliceSpec) && (Exts::rank() == 1), int> = 0>
inline constexpr auto slice(
    const std::experimental::mdspan<ET, Exts, LP, AP>& v,
    SliceSpec&& rows) noexcept
{
    return std::experimental::submdspan(v, std::forward<SliceSpec>(rows));
}

// Extract a diagonal from a matrix
template <class ET,
          class Exts,
          class LP,
          class AP,
          std::enable_if_t<
              /* Requires: */
              LP::template mapping<Exts>::is_always_strided(),
              bool> = true>
inline constexpr auto diag(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
                           int diagIdx = 0)
{
    using std::abs;
    using std::array;
    using std::min;
    using std::experimental::layout_stride;

    using size_type =
        typename std::experimental::mdspan<ET, Exts, LP, AP>::size_type;
    using extents_t = std::experimental::dextents<size_type, 1>;
    using mapping = typename layout_stride::template mapping<extents_t>;

    // mdspan components
    auto ptr = A.accessor().offset(A.data(), (diagIdx >= 0)
                                                 ? A.mapping()(0, diagIdx)
                                                 : A.mapping()(-diagIdx, 0));
    auto map = mapping(
        extents_t(
            (diagIdx >= 0)
                ? min(A.extent(0) + diagIdx, A.extent(1)) - (size_type)diagIdx
                : min(A.extent(0), A.extent(1) - diagIdx) + (size_type)diagIdx),
        array<size_type, 1>{A.stride(0) + A.stride(1)});
    auto acc_pol = typename AP::offset_policy(A.accessor());

    // return
    return std::experimental::mdspan<ET, extents_t, layout_stride,
                                     typename AP::offset_policy>(
        std::move(ptr), std::move(map), std::move(acc_pol));
}

#undef isSlice

// -----------------------------------------------------------------------------
// Deduce matrix and vector type from two provided ones

namespace internal {

#ifdef TLAPACK_PREFERRED_MATRIX_MDSPAN

    #ifndef TLAPACK_EIGEN_HH
        #ifndef TLAPACK_LEGACYARRAY_HH
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) true
        #else
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) !is_legacy_type<T>
        #endif
    #else
        #ifndef TLAPACK_LEGACYARRAY_HH
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) !is_eigen_type<T>
        #else
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                (!is_eigen_type<T> && !is_legacy_type<T>)
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
        using extents_t = std::experimental::dextents<idx_t, 2>;

        using type = std::experimental::
            mdspan<T, extents_t, std::experimental::layout_stride>;
    };

    // for two types
    // should be especialized for every new vector class
    template <class matrixA_t, typename matrixB_t>
    struct vector_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<TLAPACK_USE_PREFERRED_MATRIX_TYPE(matrixA_t) ||
                                    TLAPACK_USE_PREFERRED_MATRIX_TYPE(
                                        matrixB_t),
                                int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;
        using extents_t = std::experimental::dextents<idx_t, 1>;

        using type = std::experimental::
            mdspan<T, extents_t, std::experimental::layout_stride>;
    };

    #undef TLAPACK_USE_PREFERRED_MATRIX_TYPE

#else

    // for two types
    // should be especialized for every new matrix class
    template <class matrixA_t, class matrixB_t>
    struct matrix_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<
            is_mdspan_type<matrixA_t> && is_mdspan_type<matrixB_t> &&
                std::is_same<typename matrixA_t::layout_type,
                             typename matrixB_t::layout_type>::value,
            int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;
        using extents_t = std::experimental::dextents<idx_t, 2>;

        using type = std::experimental::
            mdspan<T, extents_t, typename matrixA_t::layout_type>;
    };
    template <class matrixA_t, class matrixB_t>
    struct matrix_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<
            is_mdspan_type<matrixA_t> && is_mdspan_type<matrixB_t> &&
                !std::is_same<typename matrixA_t::layout_type,
                              typename matrixB_t::layout_type>::value,
            int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;
        using extents_t = std::experimental::dextents<idx_t, 2>;

        using type = std::experimental::
            mdspan<T, extents_t, std::experimental::layout_stride>;
    };

    // for two types
    // should be especialized for every new vector class
    template <class matrixA_t, class matrixB_t>
    struct vector_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<
            is_mdspan_type<matrixA_t> && is_mdspan_type<matrixB_t> &&
                std::is_same<typename matrixA_t::layout_type,
                             typename matrixB_t::layout_type>::value,
            int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;
        using extents_t = std::experimental::dextents<idx_t, 1>;

        using type = std::experimental::
            mdspan<T, extents_t, typename matrixA_t::layout_type>;
    };

    // for two types
    // should be especialized for every new vector class
    template <class matrixA_t, class matrixB_t>
    struct vector_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<
            is_mdspan_type<matrixA_t> && is_mdspan_type<matrixB_t> &&
                !std::is_same<typename matrixA_t::layout_type,
                              typename matrixB_t::layout_type>::value,
            int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;
        using extents_t = std::experimental::dextents<idx_t, 1>;

        using type = std::experimental::
            mdspan<T, extents_t, std::experimental::layout_stride>;
    };

#endif  // TLAPACK_PREFERRED_MATRIX
}  // namespace internal

// -----------------------------------------------------------------------------
// Cast to Legacy arrays

template <class ET,
          class Exts,
          class LP,
          class AP,
          std::enable_if_t<Exts::rank() == 2 &&
                               LP::template mapping<Exts>::is_always_strided(),
                           int> = 0>
inline constexpr auto legacy_matrix(
    const std::experimental::mdspan<ET, Exts, LP, AP>& A) noexcept
{
    using idx_t =
        typename std::experimental::mdspan<ET, Exts, LP, AP>::size_type;

    // Here we do not use layout<std::experimental::mdspan<ET, Exts, LP, AP>>
    // on purpose. This is because we want to allow legacy_matrix to be used
    // with mdspan objects where the strides are defined at runtime.
    const Layout L = (A.stride(0) == 1 && A.stride(1) >= A.extent(0))
                         ? Layout::ColMajor
                         : Layout::RowMajor;

    assert((A.stride(0) == 1 && A.stride(1) >= A.extent(0)) ||  // col major
           (A.stride(1) == 1 && A.stride(0) >= A.extent(1)));   // row major

    return legacy::matrix<ET, idx_t>{
        L, A.extent(0), A.extent(1), A.data(),
        (L == Layout::ColMajor) ? A.stride(1) : A.stride(0)};
}

template <class ET,
          class Exts,
          class LP,
          class AP,
          std::enable_if_t<Exts::rank() == 1 &&
                               LP::template mapping<Exts>::is_always_strided(),
                           int> = 0>
inline constexpr auto legacy_matrix(
    const std::experimental::mdspan<ET, Exts, LP, AP>& A) noexcept
{
    using idx_t =
        typename std::experimental::mdspan<ET, Exts, LP, AP>::size_type;
    return legacy::matrix<ET, idx_t>{Layout::ColMajor, 1, A.size(), A.data(),
                                     A.stride(0)};
}

template <class ET,
          class Exts,
          class LP,
          class AP,
          std::enable_if_t<Exts::rank() == 1 &&
                               LP::template mapping<Exts>::is_always_strided(),
                           int> = 0>
inline constexpr auto legacy_vector(
    const std::experimental::mdspan<ET, Exts, LP, AP>& A) noexcept
{
    using idx_t =
        typename std::experimental::mdspan<ET, Exts, LP, AP>::size_type;
    return legacy::vector<ET, idx_t>{A.size(), A.data(), A.stride(0)};
}

}  // namespace tlapack

#endif  // TLAPACK_MDSPAN_HH
