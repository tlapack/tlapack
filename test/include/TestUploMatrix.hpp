/// @file test/include/TestUploMatrix.hpp
/// @brief TestUploMatrix class
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TEST_TESTUPLOMATRIX_HH
#define TLAPACK_TEST_TESTUPLOMATRIX_HH

#include <tlapack/plugins/legacyArray.hpp>

namespace tlapack {

/**
 * @brief TestUploMatrix class
 *
 * This class is used to test if a method is accessing the correct elements of a
 * matrix. The access region can be either the upper or lower triangular part of
 * the matrix.
 *
 * @tparam T Type of the elements.
 * @tparam idx_t Type of the indices.
 * @tparam uplo Uplo value.
 * @tparam L Layout value.
 *
 * @ingroup auxiliary
 */
template <class T,
          class idx_t = std::size_t,
          Uplo uplo = Uplo::Upper,
          Layout L = Layout::ColMajor>
struct TestUploMatrix : public legacyMatrix<T, idx_t, L> {
    int modifier =
        0;  ///< Modifier to the access region. Enables slicing of the matrix.

    TestUploMatrix(const legacyMatrix<T, idx_t, L>& A)
        : legacyMatrix<T, idx_t, L>(A)
    {}

    // Overload of the access operator
    inline constexpr T& operator()(idx_t i, idx_t j) const noexcept
    {
        if (uplo == Uplo::Upper)
            assert((int)i <= (int)j + modifier);
        else if (uplo == Uplo::Lower)
            assert((int)i >= (int)j + modifier);

        return legacyMatrix<T, idx_t, L>::operator()(i, j);
    };
};

// Block access specialization for TestUploMatrix

#define isSlice(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value

template <
    typename T,
    class idx_t,
    Uplo uplo,
    Layout layout,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
                            int>::type = 0>
inline constexpr auto slice(const TestUploMatrix<T, idx_t, uplo, layout>& A,
                            SliceSpecRow&& rows,
                            SliceSpecCol&& cols) noexcept
{
    TestUploMatrix<T, idx_t, uplo, layout> B(
        slice(legacyMatrix<T, idx_t, layout>(A), rows, cols));
    B.modifier = A.modifier + cols.first - rows.first;
    return B;
}

#undef isSlice

template <typename T, class idx_t, Uplo uplo, Layout layout, class SliceSpec>
inline constexpr auto rows(const TestUploMatrix<T, idx_t, uplo, layout>& A,
                           SliceSpec&& rows) noexcept
{
    TestUploMatrix<T, idx_t, uplo, layout> B(
        rows(legacyMatrix<T, idx_t, layout>(A), rows));
    B.modifier = A.modifier - rows.first;
    return B;
}

template <typename T, class idx_t, Uplo uplo, Layout layout, class SliceSpec>
inline constexpr auto cols(const TestUploMatrix<T, idx_t, uplo, layout>& A,
                           SliceSpec&& cols) noexcept
{
    TestUploMatrix<T, idx_t, uplo, layout> B(
        cols(legacyMatrix<T, idx_t, layout>(A), cols));
    B.modifier = A.modifier + cols.first;
    return B;
}

}  // namespace tlapack

#endif  // TLAPACK_TEST_TESTUPLOMATRIX_HH