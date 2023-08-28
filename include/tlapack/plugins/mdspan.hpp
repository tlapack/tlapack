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

namespace tlapack {

// -----------------------------------------------------------------------------
// Helpers

namespace mdspan {
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
}  // namespace mdspan

// -----------------------------------------------------------------------------
// Data traits

namespace traits {
    /// Layout for mdspan
    template <class ET, class Exts, class AP>
    struct layout_trait<
        std::experimental::mdspan<ET, Exts, std::experimental::layout_left, AP>,
        std::enable_if_t<Exts::rank() == 2, int>> {
        static constexpr Layout value = Layout::ColMajor;
    };
    template <class ET, class Exts, class AP>
    struct layout_trait<
        std::experimental::
            mdspan<ET, Exts, std::experimental::layout_right, AP>,
        std::enable_if_t<Exts::rank() == 2, int>> {
        static constexpr Layout value = Layout::RowMajor;
    };
    template <class ET, class Exts, class LP, class AP>
    struct layout_trait<
        std::experimental::mdspan<ET, Exts, LP, AP>,
        std::enable_if_t<(Exts::rank() == 1) &&
                             LP::template mapping<Exts>::is_always_strided(),
                         int>> {
        static constexpr Layout value = Layout::Strided;
    };

    template <class ET, class Exts, class LP, class AP>
    struct real_type_traits<std::experimental::mdspan<ET, Exts, LP, AP>, int> {
        using type = std::experimental::mdspan<real_type<ET>, Exts, LP, AP>;
    };

    template <class ET, class Exts, class LP, class AP>
    struct complex_type_traits<std::experimental::mdspan<ET, Exts, LP, AP>,
                               int> {
        using type = std::experimental::mdspan<complex_type<ET>, Exts, LP, AP>;
    };

    /// Create mdspan @see Create
    template <class ET, class Exts, class LP, class AP>
    struct CreateFunctor<std::experimental::mdspan<ET, Exts, LP, AP>,
                         std::enable_if_t<Exts::rank() == 1, int>> {
        using idx_t =
            typename std::experimental::mdspan<ET, Exts, LP>::size_type;
        using extents_t = std::experimental::dextents<idx_t, 1>;

        template <class T>
        constexpr auto operator()(std::vector<T>& v, idx_t n) const
        {
            assert(n >= 0);
            v.resize(n);  // Allocates space in memory
            return std::experimental::mdspan<T, extents_t>(v.data(), n);
        }
    };

    /// Create mdspan @see Create
    template <class ET, class Exts, class LP, class AP>
    struct CreateFunctor<std::experimental::mdspan<ET, Exts, LP, AP>,
                         std::enable_if_t<Exts::rank() == 2, int>> {
        using idx_t =
            typename std::experimental::mdspan<ET, Exts, LP>::size_type;
        using extents_t = std::experimental::dextents<idx_t, 2>;

        template <class T>
        constexpr auto operator()(std::vector<T>& v, idx_t m, idx_t n) const
        {
            assert(m >= 0 && n >= 0);
            v.resize(m * n);  // Allocates space in memory
            return std::experimental::mdspan<T, extents_t>(v.data(), m, n);
        }
    };
}  // namespace traits

// -----------------------------------------------------------------------------
// Data descriptors

// Size
template <class ET, class Exts, class LP, class AP>
constexpr auto size(const std::experimental::mdspan<ET, Exts, LP, AP>& x)
{
    return x.size();
}
// Number of rows
template <class ET, class Exts, class LP, class AP>
constexpr auto nrows(const std::experimental::mdspan<ET, Exts, LP, AP>& x)
{
    return x.extent(0);
}
// Number of columns
template <class ET, class Exts, class LP, class AP>
constexpr auto ncols(const std::experimental::mdspan<ET, Exts, LP, AP>& x)
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
constexpr auto slice(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
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
constexpr auto rows(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
                    SliceSpec&& rows) noexcept
{
    return std::experimental::submdspan(A, std::forward<SliceSpec>(rows),
                                        std::experimental::full_extent);
}

// Row
template <class ET, class Exts, class LP, class AP>
constexpr auto row(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
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
constexpr auto cols(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
                    SliceSpec&& cols) noexcept
{
    return std::experimental::submdspan(A, std::experimental::full_extent,
                                        std::forward<SliceSpec>(cols));
}

// Column
template <class ET, class Exts, class LP, class AP>
constexpr auto col(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
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
constexpr auto slice(const std::experimental::mdspan<ET, Exts, LP, AP>& v,
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
constexpr auto diag(const std::experimental::mdspan<ET, Exts, LP, AP>& A,
                    int diagIdx = 0)
{
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

// Transpose View
template <class ET, class Exts, class AP>
constexpr auto transpose_view(
    const std::experimental::
        mdspan<ET, Exts, std::experimental::layout_left, AP>& A) noexcept
{
    using matrix_t =
        std::experimental::mdspan<ET, Exts, std::experimental::layout_left, AP>;
    using idx_t = typename matrix_t::size_type;
    using extents_t =
        std::experimental::extents<idx_t, matrix_t::static_extent(1),
                                   matrix_t::static_extent(0)>;

    using std::experimental::layout_right;
    using mapping_t = typename layout_right::template mapping<extents_t>;

    mapping_t map(extents_t(A.extent(1), A.extent(0)));
    return std::experimental::mdspan<ET, extents_t, layout_right, AP>(
        A.data(), std::move(map));
}
template <class ET, class Exts, class AP>
constexpr auto transpose_view(
    const std::experimental::
        mdspan<ET, Exts, std::experimental::layout_right, AP>& A) noexcept
{
    using matrix_t =
        std::experimental::mdspan<ET, Exts, std::experimental::layout_right,
                                  AP>;
    using idx_t = typename matrix_t::size_type;
    using extents_t =
        std::experimental::extents<idx_t, matrix_t::static_extent(1),
                                   matrix_t::static_extent(0)>;

    using std::experimental::layout_left;
    using mapping_t = typename layout_left::template mapping<extents_t>;

    mapping_t map(extents_t(A.extent(1), A.extent(0)));
    return std::experimental::mdspan<ET, extents_t, layout_left, AP>(
        A.data(), std::move(map));
}
template <class ET, class Exts, class AP>
constexpr auto transpose_view(
    const std::experimental::
        mdspan<ET, Exts, std::experimental::layout_stride, AP>& A) noexcept
{
    using matrix_t =
        std::experimental::mdspan<ET, Exts, std::experimental::layout_stride,
                                  AP>;
    using idx_t = typename matrix_t::size_type;
    using extents_t =
        std::experimental::extents<idx_t, matrix_t::static_extent(1),
                                   matrix_t::static_extent(0)>;

    using std::experimental::layout_stride;
    using mapping_t = typename layout_stride::template mapping<extents_t>;

    mapping_t map(extents_t(A.extent(1), A.extent(0)),
                  std::array<idx_t, 2>{A.stride(1), A.stride(0)});
    return std::experimental::mdspan<ET, extents_t, layout_stride, AP>(
        A.data(), std::move(map));
}

// Reshape
template <
    class ET,
    class Exts,
    class LP,
    class AP,
    std::enable_if_t<Exts::rank() == 2 &&
                         (std::is_same_v<LP, std::experimental::layout_right> ||
                          std::is_same_v<LP, std::experimental::layout_left>),
                     int> = 0>
auto reshape(std::experimental::mdspan<ET, Exts, LP, AP>& A,
             std::size_t m,
             std::size_t n) noexcept
{
    using size_type =
        typename std::experimental::mdspan<ET, Exts, LP, AP>::size_type;
    using extents_t = std::experimental::dextents<size_type, 2>;
    using matrix_t = std::experimental::mdspan<ET, extents_t, LP, AP>;
    using mapping_t = typename LP::template mapping<extents_t>;

    assert((m * n == A.size()) &&
           "reshape: new shape must have the same "
           "number of elements as the original one");

    return matrix_t(A.data(), mapping_t(extents_t(m, n)));
}
template <class ET, class Exts, class AP>
auto reshape(
    std::experimental::mdspan<ET, Exts, std::experimental::layout_stride, AP>&
        A,
    std::size_t m,
    std::size_t n) noexcept
{
    using LP = std::experimental::layout_stride;
    using idx_t =
        typename std::experimental::mdspan<ET, Exts, LP, AP>::size_type;
    using extents_t = std::experimental::dextents<idx_t, 2>;
    using matrix_t = std::experimental::mdspan<ET, extents_t, LP, AP>;
    using mapping_t = typename LP::template mapping<extents_t>;

    if (m == A.extent(0) && n == A.extent(1))
        return matrix_t(A.data(), mapping_t(extents_t(m, n),
                                            std::array<idx_t, 2>{A.stride(0),
                                                                 A.stride(1)}));
    else {
        assert((m * n == A.size()) &&
               "reshape: new shape must have the same "
               "number of elements as the original one");
        assert(((A.stride(0) == 1 &&
                 (A.stride(1) == A.extent(0) || A.extent(1) <= 1)) ||
                (A.stride(1) == 1 &&
                 (A.stride(0) == A.extent(1) || A.extent(0) <= 1))) &&
               "reshape: data must be contiguous in memory");

        return matrix_t(A.data(), mapping_t(extents_t(m, n),
                                            (A.stride(0) == 1)
                                                ? std::array<idx_t, 2>{1, m}
                                                : std::array<idx_t, 2>{n, 1}));
    }
}

#undef isSlice

// -----------------------------------------------------------------------------
// Deduce matrix and vector type from two provided ones

namespace traits {

#ifdef TLAPACK_PREFERRED_MATRIX_MDSPAN

    #ifndef TLAPACK_EIGEN_HH
        #ifndef TLAPACK_LEGACYARRAY_HH
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) true
        #else
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                !legacy::is_legacy_type<T>
        #endif
    #else
        #ifndef TLAPACK_LEGACYARRAY_HH
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                !eigen::is_eigen_type<T>
        #else
            #define TLAPACK_USE_PREFERRED_MATRIX_TYPE(T) \
                (!eigen::is_eigen_type<T> && !legacy::is_legacy_type<T>)
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
            mdspan::is_mdspan_type<matrixA_t> &&
                mdspan::is_mdspan_type<matrixB_t> &&
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
            mdspan::is_mdspan_type<matrixA_t> &&
                mdspan::is_mdspan_type<matrixB_t> &&
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
            mdspan::is_mdspan_type<matrixA_t> &&
                mdspan::is_mdspan_type<matrixB_t> &&
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
            mdspan::is_mdspan_type<matrixA_t> &&
                mdspan::is_mdspan_type<matrixB_t> &&
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
}  // namespace traits

// -----------------------------------------------------------------------------
// Cast to Legacy arrays

template <class ET,
          class Exts,
          class LP,
          class AP,
          std::enable_if_t<Exts::rank() == 2 &&
                               LP::template mapping<Exts>::is_always_strided(),
                           int> = 0>
constexpr auto legacy_matrix(
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

    return legacy::Matrix<ET, idx_t>{
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
constexpr auto legacy_matrix(
    const std::experimental::mdspan<ET, Exts, LP, AP>& A) noexcept
{
    using idx_t =
        typename std::experimental::mdspan<ET, Exts, LP, AP>::size_type;
    return legacy::Matrix<ET, idx_t>{Layout::ColMajor, 1, A.size(), A.data(),
                                     A.stride(0)};
}

template <class ET,
          class Exts,
          class LP,
          class AP,
          std::enable_if_t<Exts::rank() == 1 &&
                               LP::template mapping<Exts>::is_always_strided(),
                           int> = 0>
constexpr auto legacy_vector(
    const std::experimental::mdspan<ET, Exts, LP, AP>& A) noexcept
{
    using idx_t =
        typename std::experimental::mdspan<ET, Exts, LP, AP>::size_type;
    return legacy::Vector<ET, idx_t>{A.size(), A.data(), A.stride(0)};
}

}  // namespace tlapack

#endif  // TLAPACK_MDSPAN_HH
