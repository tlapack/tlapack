/// @file plugins/starpu.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_HH
#define TLAPACK_STARPU_HH

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/starpu/Matrix.hpp"

namespace tlapack {

// Forward declarations
template <class T, std::enable_if_t<is_real<T>, int> = 0>
constexpr real_type<T> real(const T& x) noexcept;
template <class T, std::enable_if_t<is_real<T>, int> = 0>
constexpr real_type<T> imag(const T& x) noexcept;
template <class T, std::enable_if_t<is_real<T>, int> = 0>
constexpr T conj(const T& x) noexcept;

template <class T>
constexpr real_type<T> real(const starpu::MatrixEntry<T>& x) noexcept
{
    return real(T(x));
}

template <class T>
constexpr real_type<T> imag(const starpu::MatrixEntry<T>& x) noexcept
{
    return imag(T(x));
}

template <class T>
constexpr T conj(const starpu::MatrixEntry<T>& x) noexcept
{
    return conj(T(x));
}

namespace traits {
    template <class T>
    struct real_type_traits<starpu::MatrixEntry<T>, int>
        : public real_type_traits<T, int> {};
    template <class T>
    struct complex_type_traits<starpu::MatrixEntry<T>, int>
        : public complex_type_traits<T, int> {};
}  // namespace traits

}  // namespace tlapack

namespace tlapack {

// -----------------------------------------------------------------------------
// Data descriptors

// Number of rows
template <class T>
constexpr auto nrows(const starpu::Matrix<T>& A) noexcept
{
    return A.nrows();
}
// Number of columns
template <class T>
constexpr auto ncols(const starpu::Matrix<T>& A) noexcept
{
    return A.ncols();
}
// Size
template <class T>
constexpr auto size(const starpu::Matrix<T>& A) noexcept
{
    return A.nrows() * A.ncols();
}

// -----------------------------------------------------------------------------
// Block operations for starpu::Matrix

#define isSlice(SliceSpec)         \
    std::is_convertible<SliceSpec, \
                        std::tuple<starpu::idx_t, starpu::idx_t>>::value

template <
    class T,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
                            int>::type = 0>
constexpr auto slice(const starpu::Matrix<T>& A,
                     SliceSpecRow&& rows,
                     SliceSpecCol&& cols) noexcept
{
    using starpu::idx_t;

    const idx_t row0 = std::get<0>(rows);
    const idx_t col0 = std::get<0>(cols);
    const idx_t row1 = std::get<1>(rows);
    const idx_t col1 = std::get<1>(cols);

    return A.map_to_const_tiles(row0, row1, col0, col1);
}
template <
    class T,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
                            int>::type = 0>
constexpr auto slice(starpu::Matrix<T>& A,
                     SliceSpecRow&& rows,
                     SliceSpecCol&& cols) noexcept
{
    using starpu::idx_t;

    const idx_t row0 = std::get<0>(rows);
    const idx_t col0 = std::get<0>(cols);
    const idx_t row1 = std::get<1>(rows);
    const idx_t col1 = std::get<1>(cols);

    return A.map_to_tiles(row0, row1, col0, col1);
}

#undef isSlice

template <class T, class SliceSpec>
constexpr auto slice(const starpu::Matrix<T>& v,
                     SliceSpec&& range,
                     starpu::idx_t colIdx) noexcept
{
    return slice(v, std::forward<SliceSpec>(range),
                 std::make_tuple(colIdx, colIdx + 1));
}
template <class T, class SliceSpec>
constexpr auto slice(starpu::Matrix<T>& v,
                     SliceSpec&& range,
                     starpu::idx_t colIdx) noexcept
{
    return slice(v, std::forward<SliceSpec>(range),
                 std::make_tuple(colIdx, colIdx + 1));
}

template <class T, class SliceSpec>
constexpr auto slice(const starpu::Matrix<T>& v,
                     starpu::idx_t rowIdx,
                     SliceSpec&& range) noexcept
{
    return slice(v, std::make_tuple(rowIdx, rowIdx + 1),
                 std::forward<SliceSpec>(range));
}
template <class T, class SliceSpec>
constexpr auto slice(starpu::Matrix<T>& v,
                     starpu::idx_t rowIdx,
                     SliceSpec&& range) noexcept
{
    return slice(v, std::make_tuple(rowIdx, rowIdx + 1),
                 std::forward<SliceSpec>(range));
}

template <class T, class SliceSpec>
constexpr auto slice(const starpu::Matrix<T>& v, SliceSpec&& range) noexcept
{
    assert((v.nrows() <= 1 || v.ncols() <= 1) && "Matrix is not a vector");

    if (v.nrows() > 1)
        return slice(v, std::forward<SliceSpec>(range), std::make_tuple(0, 1));
    else
        return slice(v, std::make_tuple(0, 1), std::forward<SliceSpec>(range));
}
template <class T, class SliceSpec>
constexpr auto slice(starpu::Matrix<T>& v, SliceSpec&& range) noexcept
{
    assert((v.nrows() <= 1 || v.ncols() <= 1) && "Matrix is not a vector");

    if (v.nrows() > 1)
        return slice(v, std::forward<SliceSpec>(range), std::make_tuple(0, 1));
    else
        return slice(v, std::make_tuple(0, 1), std::forward<SliceSpec>(range));
}

template <class T>
constexpr auto col(const starpu::Matrix<T>& A, starpu::idx_t colIdx) noexcept
{
    return slice(A, std::make_tuple(0, A.nrows()),
                 std::make_tuple(colIdx, colIdx + 1));
}
template <class T>
constexpr auto col(starpu::Matrix<T>& A, starpu::idx_t colIdx) noexcept
{
    return slice(A, std::make_tuple(0, A.nrows()),
                 std::make_tuple(colIdx, colIdx + 1));
}

template <class T, class SliceSpec>
constexpr auto cols(const starpu::Matrix<T>& A, SliceSpec&& cols) noexcept
{
    return slice(A, std::make_tuple(0, A.nrows()),
                 std::forward<SliceSpec>(cols));
}
template <class T, class SliceSpec>
constexpr auto cols(starpu::Matrix<T>& A, SliceSpec&& cols) noexcept
{
    return slice(A, std::make_tuple(0, A.nrows()),
                 std::forward<SliceSpec>(cols));
}

template <class T>
constexpr auto row(const starpu::Matrix<T>& A, starpu::idx_t rowIdx) noexcept
{
    return slice(A, std::make_tuple(rowIdx, rowIdx + 1),
                 std::make_tuple(0, A.ncols()));
}
template <class T>
constexpr auto row(starpu::Matrix<T>& A, starpu::idx_t rowIdx) noexcept
{
    return slice(A, std::make_tuple(rowIdx, rowIdx + 1),
                 std::make_tuple(0, A.ncols()));
}

template <class T, class SliceSpec>
constexpr auto rows(const starpu::Matrix<T>& A, SliceSpec&& rows) noexcept
{
    return slice(A, std::forward<SliceSpec>(rows),
                 std::make_tuple(0, A.ncols()));
}
template <class T, class SliceSpec>
constexpr auto rows(starpu::Matrix<T>& A, SliceSpec&& rows) noexcept
{
    return slice(A, std::forward<SliceSpec>(rows),
                 std::make_tuple(0, A.ncols()));
}

template <class T>
constexpr auto diag(const starpu::Matrix<T>& A, int diagIdx = 0)
{
    throw std::runtime_error("Not implemented");
    return row(A, 0);
}
template <class T>
constexpr auto diag(starpu::Matrix<T>& A, int diagIdx = 0)
{
    throw std::runtime_error("Not implemented");
    return row(A, 0);
}

namespace traits {

    template <class TA, class TB>
    struct matrix_type_traits<starpu::Matrix<TA>, starpu::Matrix<TB>, int> {
        using T = scalar_type<TA, TB>;
        using type = starpu::Matrix<T>;
    };

    template <class TA, class TB>
    struct vector_type_traits<starpu::Matrix<TA>, starpu::Matrix<TB>, int> {
        using T = scalar_type<TA, TB>;
        using type = starpu::Matrix<T>;
    };

    /// Create starpu::Matrix<T> @see Create
    template <class U>
    struct CreateFunctor<starpu::Matrix<U>, int> {
        template <class T>
        constexpr auto operator()(std::vector<T>& v,
                                  starpu::idx_t m,
                                  starpu::idx_t n = 1) const
        {
            assert(m >= 0 && n >= 0);
            v.resize(m * n);  // Allocates space in memory
            starpu_memory_pin((void*)v.data(), m * n * sizeof(T));
            return starpu::Matrix<T>(v.data(), m, n, 1, 1);
        }
    };
}  // namespace traits

}  // namespace tlapack

#endif  // TLAPACK_STARPU_HH
