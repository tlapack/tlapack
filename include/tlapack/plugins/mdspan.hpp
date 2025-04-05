/// @file mdspan.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MDSPAN_HH
#define TLAPACK_MDSPAN_HH

#include <cassert>
#include <experimental/mdspan>  // Use mdspan from
                                // https://github.com/kokkos/mdspan because we
                                // need the `submdspan` functionality

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/plugins/stdvector.hpp"

namespace tlapack {

// -----------------------------------------------------------------------------
// Helpers

namespace traits {
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
}  // namespace traits

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

    /// Create mdspan @see CreateStatic
    template <class ET, class Exts, class LP, class AP, int n>
    struct CreateStaticFunctor<std::experimental::mdspan<ET, Exts, LP, AP>,
                               n,
                               -1,
                               std::enable_if_t<Exts::rank() == 1, int>> {
        static_assert(n >= 0);
        using idx_t =
            typename std::experimental::mdspan<ET, Exts, LP>::size_type;
        using extents_t = std::experimental::extents<idx_t, n>;

        template <typename T>
        constexpr auto operator()(T* v) const
        {
            return std::experimental::mdspan<T, extents_t>(v);
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

    /// Create mdspan @see CreateStatic
    template <class ET, class Exts, class LP, class AP, int m, int n>
    struct CreateStaticFunctor<std::experimental::mdspan<ET, Exts, LP, AP>,
                               m,
                               n,
                               std::enable_if_t<Exts::rank() == 2, int>> {
        static_assert(m >= 0 && n >= 0);
        using idx_t =
            typename std::experimental::mdspan<ET, Exts, LP>::size_type;
        using extents_t = std::experimental::extents<idx_t, m, n>;

        template <typename T>
        constexpr auto operator()(T* v) const
        {
            return std::experimental::mdspan<T, extents_t>(v);
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

// Reshape to matrix
template <
    class ET,
    class Exts,
    class LP,
    class AP,
    std::enable_if_t<(std::is_same_v<LP, std::experimental::layout_right> ||
                      std::is_same_v<LP, std::experimental::layout_left>),
                     int> = 0>
auto reshape(std::experimental::mdspan<ET, Exts, LP, AP>& A,
             std::size_t m,
             std::size_t n)
{
    using idx_t = typename std::experimental::mdspan<ET, Exts, LP>::size_type;
    using extents1_t = std::experimental::dextents<idx_t, 1>;
    using extents2_t = std::experimental::dextents<idx_t, 2>;
    using vector_t = std::experimental::mdspan<ET, extents1_t, LP>;
    using matrix_t = std::experimental::mdspan<ET, extents2_t, LP>;
    using mapping1_t = typename LP::template mapping<extents1_t>;
    using mapping2_t = typename LP::template mapping<extents2_t>;

    // constants
    const idx_t size = A.size();
    const idx_t new_size = m * n;

    // Check arguments
    if (new_size > size)
        throw std::domain_error("New size is larger than current size");

    return std::make_pair(
        matrix_t(A.data(), mapping2_t(extents2_t(m, n))),
        vector_t(A.data() + new_size, mapping1_t(extents1_t(size - new_size))));
}
template <class ET,
          class Exts,
          class AP,
          std::enable_if_t<Exts::rank() == 2, int> = 0>
auto reshape(
    std::experimental::mdspan<ET, Exts, std::experimental::layout_stride, AP>&
        A,
    std::size_t m,
    std::size_t n)
{
    using LP = std::experimental::layout_stride;
    using idx_t = typename std::experimental::mdspan<ET, Exts, LP>::size_type;
    using extents_t = std::experimental::dextents<idx_t, 2>;
    using matrix_t = std::experimental::mdspan<ET, extents_t, LP>;
    using mapping_t = typename LP::template mapping<extents_t>;

    // constants
    const idx_t size = A.size();
    const idx_t new_size = m * n;
    const bool is_contiguous =
        (size <= 1) ||
        (A.stride(0) == 1 &&
         (A.stride(1) == A.extent(0) || A.extent(1) <= 1)) ||
        (A.stride(1) == 1 && (A.stride(0) == A.extent(1) || A.extent(0) <= 1));

    // Check arguments
    if (new_size > size)
        throw std::domain_error("New size is larger than current size");
    if (A.stride(0) != 1 && A.stride(1) != 1)
        throw std::domain_error(
            "Reshaping is not available for matrices with both strides "
            "different from 1.");

    if (is_contiguous) {
        const idx_t s = size - new_size;
        if (A.stride(0) == 1)
            return std::make_pair(
                matrix_t(A.data(), mapping_t(extents_t(m, n),
                                             std::array<idx_t, 2>{1, m})),
                matrix_t(
                    A.data() + new_size,
                    mapping_t(extents_t(s, 1), std::array<idx_t, 2>{1, s})));
        else
            return std::make_pair(
                matrix_t(A.data(), mapping_t(extents_t(m, n),
                                             std::array<idx_t, 2>{n, 1})),
                matrix_t(
                    A.data() + new_size,
                    mapping_t(extents_t(1, s), std::array<idx_t, 2>{s, 1})));
    }
    else {
        std::array<idx_t, 2> strides{A.stride(0), A.stride(1)};

        if (m == A.extent(0) || n == 0) {
            return std::make_pair(
                matrix_t(A.data(), mapping_t(extents_t(m, n), strides)),
                matrix_t(A.data() + n * A.stride(1),
                         mapping_t(extents_t(m, A.extent(1) - n), strides)));
        }
        else if (n == A.extent(1) || m == 0) {
            return std::make_pair(
                matrix_t(A.data(), mapping_t(extents_t(m, n), strides)),
                matrix_t(A.data() + m * A.stride(0),
                         mapping_t(extents_t(A.extent(0) - m, n), strides)));
        }
        else {
            throw std::domain_error(
                "Cannot reshape to non-contiguous matrix if the number of rows "
                "and "
                "columns are different.");
        }
    }
}
template <class ET,
          class Exts,
          class AP,
          std::enable_if_t<Exts::rank() == 1, int> = 0>
auto reshape(
    std::experimental::mdspan<ET, Exts, std::experimental::layout_stride, AP>&
        v,
    std::size_t m,
    std::size_t n)
{
    using LP = std::experimental::layout_stride;
    using idx_t = typename std::experimental::mdspan<ET, Exts, LP>::size_type;
    using extents1_t = std::experimental::dextents<idx_t, 1>;
    using extents2_t = std::experimental::dextents<idx_t, 2>;
    using vector_t = std::experimental::mdspan<ET, extents1_t, LP>;
    using matrix_t = std::experimental::mdspan<ET, extents2_t, LP>;
    using mapping1_t = typename LP::template mapping<extents1_t>;
    using mapping2_t = typename LP::template mapping<extents2_t>;

    // constants
    const idx_t size = v.size();
    const idx_t new_size = m * n;
    const idx_t s = size - new_size;
    const idx_t stride = v.stride(0);
    const bool is_contiguous = (size <= 1 || stride == 1);

    // Check arguments
    if (new_size > size)
        throw std::domain_error("New size is larger than current size");
    if (!is_contiguous && m > 1 && n > 1)
        throw std::domain_error(
            "New sizes are not compatible with the current vector.");

    if (is_contiguous) {
        return std::make_pair(
            matrix_t(v.data(),
                     mapping2_t(extents2_t(m, n), std::array<idx_t, 2>{1, m})),
            vector_t(v.data() + new_size,
                     mapping1_t(extents1_t(s), std::array<idx_t, 1>{1})));
    }
    else {
        return std::make_pair(
            matrix_t(v.data(),
                     mapping2_t(extents2_t(m, n),
                                (m <= 1) ? std::array<idx_t, 2>{1, stride}
                                         : std::array<idx_t, 2>{stride, 1})),
            vector_t(v.data() + new_size * stride,
                     mapping1_t(extents1_t(s), std::array<idx_t, 1>{stride})));
    }
}

// Reshape to vector
template <
    class ET,
    class Exts,
    class LP,
    class AP,
    std::enable_if_t<(std::is_same_v<LP, std::experimental::layout_right> ||
                      std::is_same_v<LP, std::experimental::layout_left>),
                     int> = 0>
auto reshape(std::experimental::mdspan<ET, Exts, LP, AP>& A, std::size_t n)
{
    using idx_t = typename std::experimental::mdspan<ET, Exts, LP>::size_type;
    using extents_t = std::experimental::dextents<idx_t, 1>;
    using vector_t = std::experimental::mdspan<ET, extents_t, LP>;
    using mapping_t = typename LP::template mapping<extents_t>;

    // constants
    const idx_t size = A.size();

    // Check arguments
    if (n > size)
        throw std::domain_error("New size is larger than current size");

    return std::make_pair(
        vector_t(A.data(), mapping_t(extents_t(n))),
        vector_t(A.data() + n, mapping_t(extents_t(size - n))));
}
template <class ET,
          class Exts,
          class AP,
          std::enable_if_t<Exts::rank() == 2, int> = 0>
auto reshape(
    std::experimental::mdspan<ET, Exts, std::experimental::layout_stride, AP>&
        A,
    std::size_t n)
{
    using LP = std::experimental::layout_stride;
    using idx_t = typename std::experimental::mdspan<ET, Exts, LP>::size_type;
    using extents1_t = std::experimental::dextents<idx_t, 1>;
    using extents2_t = std::experimental::dextents<idx_t, 2>;
    using vector_t = std::experimental::mdspan<ET, extents1_t, LP>;
    using matrix_t = std::experimental::mdspan<ET, extents2_t, LP>;
    using mapping1_t = typename LP::template mapping<extents1_t>;
    using mapping2_t = typename LP::template mapping<extents2_t>;

    // constants
    const idx_t size = A.size();
    const idx_t s = size - n;
    const bool is_contiguous =
        (size <= 1) ||
        (A.stride(0) == 1 &&
         (A.stride(1) == A.extent(0) || A.extent(1) <= 1)) ||
        (A.stride(1) == 1 && (A.stride(0) == A.extent(1) || A.extent(0) <= 1));

    // Check arguments
    if (n > size)
        throw std::domain_error("New size is larger than current size");
    if (A.stride(0) != 1 && A.stride(1) != 1)
        throw std::domain_error(
            "Reshaping is not available for matrices with both strides "
            "different from 1.");

    if (is_contiguous) {
        return std::make_pair(
            vector_t(A.data(),
                     mapping1_t(extents1_t(n), std::array<idx_t, 1>{1})),
            matrix_t(
                A.data() + n,
                (A.stride(0) == 1)
                    ? mapping2_t(extents2_t(s, 1), std::array<idx_t, 2>{1, s})
                    : mapping2_t(extents2_t(1, s),
                                 std::array<idx_t, 2>{s, 1})));
    }
    else {
        std::array<idx_t, 2> strides{A.stride(0), A.stride(1)};

        if (n == 0) {
            return std::make_pair(
                vector_t(A.data(),
                         mapping1_t(extents1_t(0), std::array<idx_t, 1>{1})),
                matrix_t(A.data(), mapping2_t(A.extents(), strides)));
        }
        else if (n == A.extent(0)) {
            return std::make_pair(
                vector_t(A.data(),
                         mapping1_t(extents1_t(n),
                                    std::array<idx_t, 1>{A.stride(0)})),
                matrix_t(A.data() + A.stride(1),
                         mapping2_t(extents2_t(A.extent(0), A.extent(1) - 1),
                                    strides)));
        }
        else if (n == A.extent(1)) {
            return std::make_pair(
                vector_t(A.data(),
                         mapping1_t(extents1_t(n),
                                    std::array<idx_t, 1>{A.stride(1)})),
                matrix_t(A.data() + A.stride(0),
                         mapping2_t(extents2_t(A.extent(0) - 1, A.extent(1)),
                                    strides)));
        }
        else {
            throw std::domain_error(
                "Cannot reshape to non-contiguous matrix if the number of rows "
                "and "
                "columns are different.");
        }
    }
}
template <class ET,
          class Exts,
          class AP,
          std::enable_if_t<Exts::rank() == 1, int> = 0>
auto reshape(
    std::experimental::mdspan<ET, Exts, std::experimental::layout_stride, AP>&
        v,
    std::size_t n)
{
    using LP = std::experimental::layout_stride;
    using idx_t = typename std::experimental::mdspan<ET, Exts, LP>::size_type;
    using extents_t = std::experimental::dextents<idx_t, 1>;
    using vector_t = std::experimental::mdspan<ET, extents_t, LP>;
    using mapping_t = typename LP::template mapping<extents_t>;

    // constants
    const std::array<idx_t, 1> stride{v.stride(0)};

    // Check arguments
    if (n > v.size())
        throw std::domain_error("New size is larger than current size");

    return std::make_pair(
        vector_t(v.data(), mapping_t(extents_t(n), stride)),
        vector_t(v.data() + n, mapping_t(extents_t(v.size() - n), stride)));
}

#undef isSlice

// -----------------------------------------------------------------------------
// Deduce matrix and vector type from two provided ones

namespace traits {

    template <typename T>
    constexpr bool cast_to_mdspan_type =
        is_mdspan_type<T> || is_stdvector_type<T>
#ifdef TLAPACK_EIGEN_HH
        || is_eigen_type<T>
#endif
#ifdef TLAPACK_LEGACYARRAY_HH
        || is_legacy_type<T>
#endif
        ;

    // for two types
    // should be especialized for every new matrix class
    template <class matrixA_t, class matrixB_t>
    struct matrix_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<is_mdspan_type<matrixA_t> &&
                                    is_mdspan_type<matrixB_t> &&
                                    (layout<matrixA_t> == layout<matrixB_t>),
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

            (is_mdspan_type<matrixA_t> && is_mdspan_type<matrixB_t> &&
             !(layout<matrixA_t> == layout<matrixB_t>))

                ||

                ((is_mdspan_type<matrixA_t> || is_mdspan_type<matrixB_t>)&&(
                     !is_mdspan_type<matrixA_t> ||
                     !is_mdspan_type<
                         matrixB_t>)&&cast_to_mdspan_type<matrixA_t> &&
                 cast_to_mdspan_type<matrixB_t>),
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
        typename std::enable_if<is_mdspan_type<matrixA_t> &&
                                    is_mdspan_type<matrixB_t> &&
                                    (layout<matrixA_t> == layout<matrixB_t>),
                                int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;
        using extents_t = std::experimental::dextents<idx_t, 1>;

        using type = std::experimental::
            mdspan<T, extents_t, typename matrixA_t::layout_type>;
    };
    template <class matrixA_t, class matrixB_t>
    struct vector_type_traits<
        matrixA_t,
        matrixB_t,
        typename std::enable_if<

            (is_mdspan_type<matrixA_t> && is_mdspan_type<matrixB_t> &&
             !(layout<matrixA_t> == layout<matrixB_t>))

                ||

                ((is_mdspan_type<matrixA_t> || is_mdspan_type<matrixB_t>)&&(
                     !is_mdspan_type<matrixA_t> ||
                     !is_mdspan_type<
                         matrixB_t>)&&cast_to_mdspan_type<matrixA_t> &&
                 cast_to_mdspan_type<matrixB_t>),
            int>::type> {
        using T = scalar_type<type_t<matrixA_t>, type_t<matrixB_t>>;
        using idx_t = size_type<matrixA_t>;
        using extents_t = std::experimental::dextents<idx_t, 1>;

        using type = std::experimental::
            mdspan<T, extents_t, std::experimental::layout_stride>;
    };

#if !defined(TLAPACK_EIGEN_HH) && !defined(TLAPACK_LEGACYARRAY_HH)
    template <class vecA_t, class vecB_t>
    struct matrix_type_traits<
        vecA_t,
        vecB_t,
        std::enable_if_t<traits::is_stdvector_type<vecA_t> &&
                             traits::is_stdvector_type<vecB_t>,
                         int>> {
        using T = scalar_type<type_t<vecA_t>, type_t<vecB_t>>;
        using extents_t = std::experimental::dextents<std::size_t, 2>;

        using type = std::experimental::
            mdspan<T, extents_t, std::experimental::layout_left>;
    };

    template <class vecA_t, class vecB_t>
    struct vector_type_traits<
        vecA_t,
        vecB_t,
        std::enable_if_t<traits::is_stdvector_type<vecA_t> &&
                             traits::is_stdvector_type<vecB_t>,
                         int>> {
        using T = scalar_type<type_t<vecA_t>, type_t<vecB_t>>;
        using extents_t = std::experimental::dextents<std::size_t, 1>;

        using type = std::experimental::
            mdspan<T, extents_t, std::experimental::layout_left>;
    };
#endif

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
