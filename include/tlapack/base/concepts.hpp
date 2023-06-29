/// @file concepts.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ARRAY_CONCEPTS_HH
#define TLAPACK_ARRAY_CONCEPTS_HH

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/types.hpp"

#if __cplusplus >= 202002L
    #include <concepts>

namespace tlapack {

// Forward declarations
template <typename T>
inline T abs(const T& x);

namespace concepts {

    using std::ceil;
    using std::floor;
    using std::isinf;
    using std::isnan;
    using std::pow;
    using std::sqrt;

    /** @interface tlapack::concepts::Arithmetic
     * @brief Concept for a type that supports arithmetic operations.
     *
     * An arithmetic type must implement the operators @c +, @c -, @c *, and
     * @c /,, with constant operands, and @c =, @c +=, @c -=, @c *=, and @c /=
     * with a constant second operand.
     *
     * @tparam T Type.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Arithmetic = requires(const T& a, const T& b, T& c)
    {
        // Arithmetic operations
        c = a + b;
        c = a - b;
        c = a * b;
        c = a / b;

        // Arithmetic operations with assignment
        c += a;
        c -= a;
        c *= a;
        c /= a;
    };

    /** @interface tlapack::concepts::Real
     * @brief Concept for real scalar types.
     *
     * A real type is an ordered type that supports arithmetic operations.
     * Moreover,
     *
     * - it must be constructible from integer and floating-points. It must also
     * be constructible from a default constructor that takes no arguments.
     *
     * - it must implement the copy assignment operator @c =.
     *
     * - it must support the math operators @c abs(), @c sqrt(), @c pow(),
     * @c ceil(), and @c floor(). Those functions must be callable from the
     * namespace @c tlapack.
     *
     * - it must support the boolean functions @c isinf() and @c isnan(), which
     * must be callable from the namespace @c tlapack.
     *
     * @tparam T Type.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Real =
        Arithmetic<T>&& std::totally_ordered<T>&& requires(const T& a, T& b)
    {
        // Constructors
        T();
        T(0);
        T(0.0);

        // Assignment
        b = a;

        // Inf and NaN checks
        {
            isinf(a)
        }
        ->std::same_as<bool>;
        {
            isnan(a)
        }
        ->std::same_as<bool>;

        // Math functions
        abs(a);
        sqrt(a);
        pow(a, 0);
        ceil(a);
        floor(a);
    };

    /** @interface tlapack::concepts::Complex
     * @brief Concept for complex scalar types.
     *
     * A complex type is a type that supports arithmetic and comparison
     * operations. Real and imaginary parts have the same type, and this must
     * satisfy the concept tlapack::concepts::Real. Moreover,
     *
     * - it must be constructible from two real values. It must also be
     * constructible from a default constructor that takes no arguments.
     *
     * - it must implement the copy assignment operator @c =. It must also
     * implement the assignment operator @c = that takes a real value as the
     * second argument.
     *
     * - it must support the accessors @c real(), @c imag(), and @c conj(). The
     * latter returns a complex type of the same type as the input. Those
     * functions must be callable from the namespace @c tlapack.
     *
     * - it must support the math function @c abs(). This function must be
     * callable from the namespace @c tlapack.
     *
     * @tparam T Type.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Complex = Arithmetic<T>&& std::equality_comparable<T>&&
        Real<real_type<T>>&& requires(const T& a, T& b)
    {
        // Constructors
        T();
        T(real_type<T>(), real_type<T>());

        // Assignment
        b = a;
        b = real(a);

        // Accessors
        {
            real(a)
        }
        ->std::same_as<real_type<T>>;
        {
            imag(a)
        }
        ->std::same_as<real_type<T>>;
        {
            conj(a)
        }
        ->std::same_as<T>;

        // Math functions
        abs(a);
    };

    /** @interface tlapack::concepts::Scalar
     * @brief Concept for scalar types.
     *
     * A scalar type is a type that supports arithmetic and comparison
     * operations. The types @c real_type<T> and @c complex_type<T> must satisfy
     * the concepts tlapack::concepts::Real and tlapack::concepts::Complex,
     * respectively. Moreover,
     *
     * - it must be move assignable.
     *
     * - it must support the math function @c abs(). This function must be
     * callable from the namespace @c tlapack.
     *
     * @tparam T Scalar type.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Scalar = Arithmetic<T>&& std::equality_comparable<T>&&
        Real<real_type<T>>&& Complex<complex_type<T>>&& requires(T a)
    {
        // Assignable from a rvalue reference
        a = (T &&) a;

        // Math functions
        abs(a);
    };

    /** @interface tlapack::concepts::Vector
     * @brief Concept for vectors.
     *
     * A vector is a one-dimensional array of elements. The indexes and sizes of
     * a vector are representable by an integral type @c idx_t. The following
     * operations must be available:
     *
     * - Entry access using @c vector_t::operator[](idx_t). Entry i of a vector
     * v is accessed using <tt>v[i]</tt>.
     *
     * - Number of entries using @c tlapack::size(vector_t). This function must
     * be callable from the namespace @c tlapack.
     *
     * @tparam vector_t Vector type.
     *
     * @ingroup concepts
     */
    template <typename vector_t>
    concept Vector = requires(const vector_t& v)
    {
        // Entry access using operator[i]
        v[0];

        // Number of entries
        {
            size(v)
        }
        ->std::integral<>;
    };

    /** @interface tlapack::concepts::SliceableVector
     * @brief Concept for a vector that can be sliced into subvectors.
     *
     * A sliceable vector is a tlapack::concepts::Vector that can be sliced into
     * subvectors. A subvector is a light-weigth view of the original vector.
     * That means that the value of the entries should not be copied. It also
     * means that any changes in the subvector will automatically reflect a
     * change in the original vector, and vice-versa. Routines in \<T\>LAPACK
     * use the following operations to slice a vector:
     *
     * - View of entries i to j, excluding j, a vector v of length n, where 0 <=
     * i <= j <= n, using the function @c slice(vector_t&, std::pair<idx_t,
     * idx_t>). The call <tt>slice(v, std::pair{i,j})</tt> returns a vector of
     * length (j-i) whose type satisfy tlapack::concepts::Vector. This function
     * must be callable from the namespace @c tlapack.
     *
     * @tparam vector_t Vector type.
     *
     * @ingroup concepts
     */
    template <typename vector_t>
    concept SliceableVector = Vector<vector_t>&& requires(const vector_t& v)
    {
        // Subvector view
        {
            slice(v, std::pair<int, int>{0, 0})
        }
        ->Vector<>;
    };

    /** @interface tlapack::concepts::Matrix
     * @brief Concept for a matrix.
     *
     * A matrix is a two-dimensional array of elements. The indexes and sizes of
     * a matrix are representable by an integral type @c idx_t. The following
     * operations must be available:
     *
     * - Entry access using @c matrix_t::operator(idx_t, idx_t). Entry i of a
     * matrix A is accessed using <tt>A(i,j)</tt>.
     *
     * - Number of rows using a function @c nrows(const matrix_t&). This
     * function must be callable from the namespace @c tlapack.
     *
     * - Number of columns using a function @c ncols(const matrix_t&). This
     * function must be callable from the namespace @c tlapack.
     *
     * @tparam matrix_t Matrix type.
     *
     * @ingroup concepts
     */
    template <typename matrix_t>
    concept Matrix = requires(const matrix_t& A)
    {
        // Entry access using operator(i,j)
        A(0, 0);

        // Number of rows
        {
            nrows(A)
        }
        ->std::integral<>;

        // Number of columns
        {
            ncols(A)
        }
        ->std::integral<>;
    };

    /** @interface tlapack::concepts::SliceableMatrix
     * @brief Concept for a matrix that can be sliced into submatrices.
     *
     * A sliceable matrix is a tlapack::concepts::Matrix that can be sliced into
     * submatrices. A submatrix is a light-weigth view of the original matrix.
     * That means that the value of the entries should not be copied. It also
     * means that any changes in the submatrix will automatically reflect a
     * change in the original matrix, and vice-versa. Routines in \<T\>LAPACK
     * use the following operations to slice a matrix:
     *
     * - Submatrix view A(i:j,k:l) of a m-by-n matrix A(0:m,0:n), where
     * 0 <= i <= j <= m and 0 <= k <= l <= n, using the function
     * @c slice(matrix_t&, std::pair<idx_t,idx_t>, std::pair<idx_t,idx_t>). The
     * call <tt>slice(A, std::pair{i,j}, std::pair{k,l})</tt> returns a
     * (j-i)-by-(l-k) matrix whose type satisfy tlapack::concepts::Matrix.
     *
     * - View of rows i to j, excluding j, of a m-by-n matrix A(0:m,0:n), where
     * 0 <= i <= j <= m, using the function @c rows(matrix_t&,
     * std::pair<idx_t,idx_t>). The call <tt>rows(A, std::pair{i,j})</tt>
     * returns a (j-i)-by-n matrix whose type satisfy tlapack::concepts::Matrix.
     *
     * - View of columns i to j, excluding j, of a m-by-n matrix A(0:m,0:n),
     * where 0 <= i <= j <= n, using the function @c cols(matrix_t&,
     * std::pair<idx_t,idx_t>). The call <tt>cols(A, std::pair{i,j})</tt>
     * returns a m-by-(j-i) matrix whose type satisfy tlapack::concepts::Matrix.
     *
     * - View of row i of a m-by-n matrix A(0:m,0:n), where 0 <= i < m, using
     * the function @c row(matrix_t&, idx_t). The call <tt>row(A, i)</tt>
     * returns a vector of length n whose type satisfy
     * tlapack::concepts::Vector.
     *
     * - View of column i of a m-by-n matrix A(0:m,0:n), where 0 <= i < n, using
     * the function @c col(matrix_t&, idx_t). The call <tt>col(A, i)</tt>
     * returns a vector of length m whose type satisfy
     * tlapack::concepts::Vector.
     *
     * - View of entries i to j, excluding j, of row k of a m-by-n matrix, where
     * 0 <= i <= j <= n and 0 <= k < m, using the function @c slice(matrix_t&,
     * idx_t, std::pair<idx_t,idx_t>). The call <tt>slice(A, k, std::pair{i,j})
     * </tt> returns a vector of length (j-i) whose type satisfy
     * tlapack::concepts::Vector.
     *
     * - View of entries i to j, excluding j, of column k of a m-by-n matrix,
     * where 0 <= i <= j <= m and 0 <= k < n, using the function
     * @c slice(matrix_t&, std::pair<idx_t,idx_t>, idx_t). The call
     * <tt>slice(A, std::pair{i,j}, k)</tt> returns a vector of length (j-i)
     * whose type satisfy tlapack::concepts::Vector.
     *
     * - View of the i-th diagonal of a m-by-n matrix A(0:m,0:n), where -m < i <
     * n, using the function @c diag(matrix_t&, idx_t = 0). The call
     * <tt>diag(A, i)</tt> returns a vector of length <tt>min(m,n)-abs(i)</tt>
     * whose type satisfy tlapack::concepts::Vector.
     *
     * @note The functions @c slice, @c rows, @c cols, @c row, @c col, and
     * @c diag are required to be callable from the namespace @c tlapack.
     *
     * @tparam matrix_t Matrix type.
     *
     * @ingroup concepts
     */
    template <typename matrix_t>
    concept SliceableMatrix = Matrix<matrix_t>&& requires(const matrix_t& A)
    {
        // Submatrix view (matrix)
        {
            slice(A, std::pair<int, int>{0, 1}, std::pair<int, int>{0, 1})
        }
        ->Matrix<>;

        // View of multiple rows (matrix)
        {
            rows(A, std::pair<int, int>{0, 1})
        }
        ->Matrix<>;

        // View of multiple columns (matrix)
        {
            cols(A, std::pair<int, int>{0, 1})
        }
        ->Matrix<>;

        // Row view (vector)
        {
            row(A, 0)
        }
        ->Vector<>;

        // Column view (vector)
        {
            col(A, 0)
        }
        ->Vector<>;

        // View of a slice of a row (vector)
        {
            slice(A, 0, std::pair<int, int>{0, 1})
        }
        ->Vector<>;

        // View of a slice of a column (vector)
        {
            slice(A, std::pair<int, int>{0, 1}, 0)
        }
        ->Vector<>;

        // Diagonal view (vector)
        {
            diag(A)
        }
        ->Vector<>;

        // Off-diagonal view (vector)
        {
            diag(A, 1)
        }
        ->Vector<>;
    };

    template <typename T>
    concept Index = std::integral<T>;

    template <typename T>
    concept Side = std::convertible_to<T, tlapack::Side>;

    template <typename T>
    concept Direction = std::convertible_to<T, tlapack::Direction>;

    template <typename T>
    concept Op = std::convertible_to<T, tlapack::Op>;

    template <typename T>
    concept StoreV = std::convertible_to<T, tlapack::StoreV>;

    template <typename T>
    concept Norm = std::convertible_to<T, tlapack::Norm>;

    template <typename T>
    concept Uplo = std::convertible_to<T, tlapack::Uplo>;

    template <typename T>
    concept Diag = std::convertible_to<T, tlapack::Diag>;

}  // namespace concepts
}  // namespace tlapack

    /// Macro for tlapack::concepts::Matrix compatible with C++17.
    #define TLAPACK_MATRIX tlapack::concepts::Matrix

    /// Macro for tlapack::concepts::SliceableMatrix compatible with C++17.
    #define TLAPACK_SMATRIX tlapack::concepts::SliceableMatrix

    /// Macro for tlapack::concepts::Vector compatible with C++17.
    #define TLAPACK_VECTOR tlapack::concepts::Vector

    /// Macro for tlapack::concepts::SliceableVector compatible with C++17.
    #define TLAPACK_SVECTOR tlapack::concepts::SliceableVector

    /// Macro for tlapack::concepts::Scalar compatible with C++17.
    #define TLAPACK_SCALAR tlapack::concepts::Scalar

    /// Macro for tlapack::concepts::Real compatible with C++17.
    #define TLAPACK_REAL tlapack::concepts::Real

    /// Macro for tlapack::concepts::Complex compatible with C++17.
    #define TLAPACK_COMPLEX tlapack::concepts::Complex

    /// Macro for tlapack::concepts::Index compatible with C++17.
    #define TLAPACK_INDEX tlapack::concepts::Index

    /// Macro for tlapack::concepts::Side compatible with C++17.
    #define TLAPACK_SIDE tlapack::concepts::Side

    /// Macro for tlapack::concepts::Direction compatible with C++17.
    #define TLAPACK_DIRECTION tlapack::concepts::Direction

    /// Macro for tlapack::concepts::Op compatible with C++17.
    #define TLAPACK_OP tlapack::concepts::Op

    /// Macro for tlapack::concepts::StoreV compatible with C++17.
    #define TLAPACK_STOREV tlapack::concepts::StoreV

    /// Macro for tlapack::concepts::Norm compatible with C++17.
    #define TLAPACK_NORM tlapack::concepts::Norm

    /// Macro for tlapack::concepts::Uplo compatible with C++17.
    #define TLAPACK_UPLO tlapack::concepts::Uplo

    /// Macro for tlapack::concepts::Diag compatible with C++17.
    #define TLAPACK_DIAG tlapack::concepts::Diag
#else
    // Concepts are a C++20 feature, so just define them as `class` for earlier
    // versions
    #define TLAPACK_MATRIX class
    #define TLAPACK_SMATRIX class

    #define TLAPACK_VECTOR class
    #define TLAPACK_SVECTOR class

    #define TLAPACK_SCALAR class
    #define TLAPACK_REAL class
    #define TLAPACK_COMPLEX class
    #define TLAPACK_INDEX class

    #define TLAPACK_SIDE class
    #define TLAPACK_DIRECTION class
    #define TLAPACK_OP class
    #define TLAPACK_STOREV class
    #define TLAPACK_NORM class
    #define TLAPACK_UPLO class
    #define TLAPACK_DIAG class

#endif

#endif  // TLAPACK_ARRAY_CONCEPTS_HH
