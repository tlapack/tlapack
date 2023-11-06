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

#if __cplusplus >= 202002L
    #include <cmath>
    #include <concepts>
    #include <cstddef>
    #include <type_traits>

    #include "tlapack/base/arrayTraits.hpp"
    #include "tlapack/base/types.hpp"

namespace tlapack {

// C++ standard math functions:
using std::abs;
using std::ceil;
using std::floor;
using std::isinf;
using std::isnan;
using std::log2;
using std::max;
using std::min;
using std::pow;  // We only use pow(int, T), see below in the concept Real.
using std::sqrt;

// C++ standard types:
using std::pair;

namespace concepts {
    /** @interface tlapack::concepts::Arithmetic
     * @brief Concept for a type that supports arithmetic operations.
     *
     * An arithmetic type must implement the operators @c +, @c -, @c *, and
     * @c /,, with constant operands, and @c =, @c +=, @c -=, @c *=, and @c /=
     * with a constant second operand. It must also implement the unary operator
     * @c - for the change of sign.
     *
     * @tparam T Type.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Arithmetic = requires(const T& a, const T& b, T& c)
    {
        // Arithmetic and assignment operations
        c = a + b;
        c = a - b;
        c = a * b;
        c = a / b;
        c = -a;

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
     * - it must support the math operators @c abs(), @c sqrt(), @c pow(int,),
     * @c log2(), @c ceil(), and @c floor(). Those functions must be callable
     * from the namespace @c tlapack.
     *
     * - it must support the boolean functions @c isinf() and @c isnan(), which
     * must be callable from the namespace @c tlapack.
     *
     * - it must implement the functions @c min(T,T) and @c max(T,T) to decide
     * the minimum and maximum in a pair of values.
     *
     * - it must have a specialization of @c std::numeric_limits<>.
     *
     * @tparam T Type.
     *
     * @note Note that the functions @c min() and @c max() may not propagate
     * NaNs. The implementation in the C++ standard library says nothing about
     * NaN propagation in @c std::min() and @c std::max() and we follow the same
     * rule. The C++ documentation shows possible implementations that do not
     * propagate NaNs. See: https://en.cppreference.com/w/cpp/algorithm/max and
     * https://en.cppreference.com/w/cpp/algorithm/min.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Real = Arithmetic<T>&& std::totally_ordered<T>&&
        std::numeric_limits<T>::is_specialized&& requires(const T& a, T& b)
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
        pow(2, a);
        log2(a);
        ceil(a);
        floor(a);
        min(a, b);
        max(a, b);
    };

    /** @interface tlapack::concepts::Complex
     * @brief Concept for complex scalar types.
     *
     * A complex type is a type that supports arithmetic and comparison
     * operations. Real and imaginary parts have the same type, and they must
     * satisfy the concept tlapack::concepts::Real. The type @c real_type<T>
     * must also satisfy the concept tlapack::concepts::Real. Moreover,
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
     * - it must support the math functions @c abs() and @c sqrt(). These
     * functions must be callable from the namespace @c tlapack.
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
        sqrt(a);
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
     * - it must support the math functions @c abs() and @c sqrt(). These
     * functions must be callable from the namespace @c tlapack.
     *
     * @tparam T Scalar type.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Scalar = Arithmetic<T>&& std::equality_comparable<T>&&
        Real<real_type<T>>&& Complex<complex_type<T>>&& requires(T&& a)
    {
        // Assignable from a rvalue reference
        a = std::forward<T>(a);

        // Math functions
        abs(a);
        sqrt(a);
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
     * - Number of entries using @c size(const vector_t&). This function must be
     * callable from the namespace @c tlapack.
     *
     * Optionally, the vector type can also implement:
     *
     * - @c tlapack::traits::layout_trait<vector_t,int>, with a member @c value
     * that satisfies the concept tlapack::concepts::Layout.
     *
     * - @c tlapack::traits::matrix_type_traits<vector_t,vector_t,int>, with a
     * member @c type that satisfies the concept tlapack::concepts::Matrix.
     *
     * - @c tlapack::traits::vector_type_traits<vector_t,vector_t,int>, with a
     * member @c type that satisfies the concept tlapack::concepts::Vector.
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
     * subvectors. A subvector is a light-weight view of the original vector.
     * That means that the value of the entries should not be copied. It also
     * means that any changes in the subvector will automatically reflect a
     * change in the original vector, and vice-versa. Routines in \<T\>LAPACK
     * use the following operations to slice a vector:
     *
     * - View of entries i to j, excluding j, a vector v of length n, where 0 <=
     * i <= j <= n, using the function @c slice(vector_t&, pair<idx_t, idx_t>).
     * The call <tt>slice(v, pair{i,j})</tt> returns a vector of length (j-i)
     * whose type satisfy tlapack::concepts::Vector. This function must be
     * callable from the namespace @c tlapack.
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
            slice(v, pair<int, int>{0, 0})
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
     * - Number of rows using a function @c nrows(const matrix_t&).
     *
     * - Number of columns using a function @c ncols(const matrix_t&).
     *
     * - Number of entries using a function @c size(const matrix_t&).
     *
     * @note The functions @c nrows, @c ncols, and @c size are required to be
     * callable from the namespace @c tlapack.
     *
     * Optionally, the matrix type can also implement:
     *
     * - @c tlapack::traits::layout_trait<matrix_t,int>, with a member @c value
     * that satisfies the concept tlapack::concepts::Layout.
     *
     * - @c tlapack::traits::matrix_type_traits<matrix_t,matrix_t>, with a
     * member @c type that satisfies the concept tlapack::concepts::Matrix.
     *
     * - @c tlapack::traits::vector_type_traits<matrix_t,matrix_t>, with a
     * member @c type that satisfies the concept tlapack::concepts::Vector.
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

        // Number of entries
        {
            size(A)
        }
        ->std::integral<>;
    };

    /** @interface tlapack::concepts::SliceableMatrix
     * @brief Concept for a matrix that can be sliced into submatrices.
     *
     * A sliceable matrix is a tlapack::concepts::Matrix that can be sliced into
     * submatrices. A submatrix is a light-weight view of the original matrix.
     * That means that the value of the entries should not be copied. It also
     * means that any changes in the submatrix will automatically reflect a
     * change in the original matrix, and vice-versa. Routines in \<T\>LAPACK
     * use the following operations to slice a matrix:
     *
     * - Submatrix view A(i:j,k:l) of a m-by-n matrix A(0:m,0:n), where
     * 0 <= i <= j <= m and 0 <= k <= l <= n, using the function
     * @c slice(matrix_t&, pair<idx_t,idx_t>, pair<idx_t,idx_t>). The
     * call <tt>slice(A, pair{i,j}, pair{k,l})</tt> returns a
     * (j-i)-by-(l-k) matrix whose type satisfy tlapack::concepts::Matrix.
     *
     * - View of rows i to j, excluding j, of a m-by-n matrix A(0:m,0:n), where
     * 0 <= i <= j <= m, using the function @c rows(matrix_t&,
     * pair<idx_t,idx_t>). The call <tt>rows(A, pair{i,j})</tt> returns a
     * (j-i)-by-n matrix whose type satisfy tlapack::concepts::Matrix.
     *
     * - View of columns i to j, excluding j, of a m-by-n matrix A(0:m,0:n),
     * where 0 <= i <= j <= n, using the function @c cols(matrix_t&,
     * pair<idx_t,idx_t>). The call <tt>cols(A, pair{i,j})</tt>
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
     * idx_t, pair<idx_t,idx_t>). The call <tt>slice(A, k, pair{i,j})
     * </tt> returns a vector of length (j-i) whose type satisfy
     * tlapack::concepts::Vector.
     *
     * - View of entries i to j, excluding j, of column k of a m-by-n matrix,
     * where 0 <= i <= j <= m and 0 <= k < n, using the function
     * @c slice(matrix_t&, pair<idx_t,idx_t>, idx_t). The call
     * <tt>slice(A, pair{i,j}, k)</tt> returns a vector of length (j-i)
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
            slice(A, pair<int, int>{0, 1}, pair<int, int>{0, 1})
        }
        ->Matrix<>;

        // View of multiple rows (matrix)
        {
            rows(A, pair<int, int>{0, 1})
        }
        ->Matrix<>;

        // View of multiple columns (matrix)
        {
            cols(A, pair<int, int>{0, 1})
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
            slice(A, 0, pair<int, int>{0, 1})
        }
        ->Vector<>;

        // View of a slice of a column (vector)
        {
            slice(A, pair<int, int>{0, 1}, 0)
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

    /** @interface tlapack::concepts::TransposableMatrix
     * @brief Concept for a matrix that can be transposed.
     *
     * A transposable matrix is a tlapack::concepts::Matrix that can be
     * transposed. Ideally, the transpose operation must not involve dynamic
     * memory allocation, so that the resulting transposed matrix is a view for
     * the same data in the original matrix. The operation
     * @c transpose_view(matrix_t&) must be callable from the namespace
     * @c tlapack.
     *
     * @tparam matrix_t Matrix type.
     *
     * @ingroup concepts
     */
    template <typename matrix_t>
    concept TransposableMatrix = Matrix<matrix_t>&& requires(const matrix_t& A)
    {
        // Transpose view
        {
            transpose_view(A)
        }
        ->Matrix<>;
    };

    // Workspace matrices

    namespace internal {
        template <typename pair_t>
        concept PairOfTransposableMatrixAndOther =
            TransposableMatrix<typename pair_t::first_type>;

        template <typename pair_t>
        concept PairOfVectorAndOther = Vector<typename pair_t::first_type>;
    }  // namespace internal

    /** @interface tlapack::concepts::Workspace
     * @brief Concept for a workspace.
     *
     * A workspace is a type that can be reshaped. The reshape operation returns
     * a @c std::pair of two objects: a matrix (or vector) and an object that
     * holds the remaining workspace. The type, layout and other properties of
     * the returned matrix (or vector) are defined by the reshape operation and,
     * therefore, each object type has should have its own implementation of
     * that functionality. Since the remaining workspace is returned, the shape
     * of the matrix (or vector) does not need to match the size of the original
     * workspace. For instance, if an object @c work has size 50, the call @c
     * reshape(work, 10, 2) returns a pair of a 10-by-2 matrix and a workspace
     * of size 30. The matrix uses 20 entries of the original workspace, and the
     * remaining 30 entries are stored in the remaining workspace. In a similar
     * example, the call
     * @c reshape(work, 35) returns a pair of a vector of size 35 and a
     * workspace of size 15. The operations
     * @c reshape(T&, size_t) and @c reshape(T&, size_t, size_t) must be
     * callable from the namespace @c tlapack.
     *
     * @note We require that the matrix returned in
     * @c reshape(T&, size_t, size_t) satisfies the concept
     * tlapack::concepts::TransposableMatrix.
     *
     * @tparam work_t Workspace type.
     *
     * @ingroup concepts
     */
    template <typename work_t>
    concept Workspace = requires(work_t& work)
    {
        // Reshape into a matrix
        {
            reshape(work, 0, 0)
        }
        ->internal::PairOfTransposableMatrixAndOther<>;

        // Reshape into a vector
        {
            reshape(work, 0)
        }
        ->internal::PairOfVectorAndOther<>;
    };

    // Other scalar concepts

    /** @interface tlapack::concepts::Index
     * @brief Concept for index types.
     *
     * An index type is an integral type, e.g., int, long, std::size_t, etc.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Index = std::integral<T>;

    /** @interface tlapack::concepts::Side
     * @brief Concept for types that represent tlapack::Side.
     *
     * @tparam T Type is implicitly and explicitly convertible to tlapack::Side.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Side = std::convertible_to<T, tlapack::Side>;

    /** @interface tlapack::concepts::Direction
     * @brief Concept for types that represent tlapack::Direction.
     *
     * @tparam T Type is implicitly and explicitly convertible to
     * tlapack::Direction.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Direction = std::convertible_to<T, tlapack::Direction>;

    /** @interface tlapack::concepts::Op
     * @brief Concept for types that represent tlapack::Op.
     *
     * @tparam T Type is implicitly and explicitly convertible to tlapack::Op.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Op = std::convertible_to<T, tlapack::Op>;

    /** @interface tlapack::concepts::StoreV
     * @brief Concept for types that represent tlapack::StoreV.
     *
     * @tparam T Type is implicitly and explicitly convertible to
     * tlapack::StoreV.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept StoreV = std::convertible_to<T, tlapack::StoreV>;

    /** @interface tlapack::concepts::Norm
     * @brief Concept for types that represent tlapack::Norm.
     *
     * @tparam T Type is implicitly and explicitly convertible to tlapack::Norm.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Norm = std::convertible_to<T, tlapack::Norm>;

    /** @interface tlapack::concepts::Uplo
     * @brief Concept for types that represent tlapack::Uplo.
     *
     * @tparam T Type is implicitly and explicitly convertible to tlapack::Uplo.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Uplo = std::convertible_to<T, tlapack::Uplo>;

    /** @interface tlapack::concepts::Diag
     * @brief Concept for types that represent tlapack::Diag.
     *
     * @tparam T Type is implicitly and explicitly convertible to tlapack::Diag.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept Diag = std::convertible_to<T, tlapack::Diag>;

    // Legacy vector and matrix types

    /** @interface tlapack::concepts::LegacyArray
     * @brief Concept for types that can be converted to a legacy matrix.
     *
     * The concept tlapack::concepts::LegacyArray applies to types that can be
     * converted to tlapack::legacy::Matrix. The conversion is performed by
     * calling @c legacy_matrix(const T&), which should be callable from the
     * namespace @c tlapack.
     *
     * @tparam T Matrix or Vector type.
     *
     * @ingroup concepts
     */
    template <typename T>
    concept LegacyArray = requires(const T& A)
    {
        {
            (legacy_matrix(A)).layout
        }
        ->std::convertible_to<tlapack::Layout>;
        {
            (legacy_matrix(A)).m
        }
        ->std::convertible_to<size_type<T>>;
        {
            (legacy_matrix(A)).n
        }
        ->std::convertible_to<size_type<T>>;
        {
            (legacy_matrix(A)).ptr[0]
        }
        ->std::convertible_to<type_t<T>>;
        {
            (legacy_matrix(A)).ldim
        }
        ->std::convertible_to<size_type<T>>;
    };

    /** @interface tlapack::concepts::LegacyMatrix
     * @brief Concept for matrices that can be converted to a legacy matrix.
     *
     * A legacy matrix is a matrix that can be converted to
     * tlapack::legacy::Matrix. The conversion is performed by calling
     * @c legacy_matrix(const T&), which should be callable from the namespace
     * @c tlapack. The layout of the matrix, tlapack::layout<matrix_t>, must be
     * either tlapack::Layout::ColMajor or tlapack::Layout::RowMajor.
     *
     * @tparam matrix_t Matrix type.
     *
     * @ingroup concepts
     */
    template <typename matrix_t>
    concept LegacyMatrix = Matrix<matrix_t>&& LegacyArray<matrix_t> &&
                           ((layout<matrix_t> == Layout::ColMajor) ||
                            (layout<matrix_t> == Layout::RowMajor));

    /** @interface tlapack::concepts::LegacyVector
     * @brief Concept for vectors that can be converted to a legacy vector.
     *
     * A legacy vector is a vector that can be converted to
     * tlapack::legacy::Vector. The conversion is performed by calling
     * @c legacy_vector(const T&), which should be callable from the namespace
     * @c tlapack. The layout of the vector, tlapack::layout<vector_t>, must be
     * either tlapack::Layout::ColMajor, tlapack::Layout::RowMajor, or
     * tlapack::Layout::Strided. Moreover, a legacy vector must also satisfy the
     * concept tlapack::concepts::LegacyArray.
     *
     * @tparam vector_t Vector type.
     *
     * @ingroup concepts
     */
    template <typename vector_t>
    concept LegacyVector = Vector<vector_t>&& LegacyArray<vector_t> &&
                           ((layout<vector_t> == Layout::ColMajor) ||
                            (layout<vector_t> == Layout::RowMajor) ||
                            (layout<vector_t> == Layout::Strided)) &&
                           requires(const vector_t& v)
    {
        {
            (legacy_vector(v)).n
        }
        ->std::convertible_to<size_type<vector_t>>;
        {
            (legacy_vector(v)).ptr[0]
        }
        ->std::convertible_to<type_t<vector_t>>;
        {
            (legacy_vector(v)).inc
        }
        ->std::convertible_to<size_type<vector_t>>;
    };

    // Matrix and vector types that can be created

    /** @interface tlapack::concepts::ConstructableArray
     * @brief Concept for arrays that implement tlapack::traits::CreateFunctor
     * and tlapack::traits::CreateStaticFunctor.
     *
     * A constructable array must provide implementations of the functors
     * tlapack::traits::CreateFunctor and tlapack::traits::CreateStaticFunctor
     * for the types tlapack::matrix_type<array_t> and
     * tlapack::vector_type<array_t>.
     *
     * - @c tlapack::traits::CreateFunctor<tlapack::matrix_type<array_t>, int>
     * must provide the method @c operator()(std::vector<T>&, idx_t, idx_t),
     * where T = tlapack::type_t<array_t> and idx_t =
     * tlapack::size_type<array_t>. The output of the functor satisfies the
     * concept tlapack::concepts::Matrix.
     *
     * - @c tlapack::traits::CreateFunctor<tlapack::vector_type<array_t>, int>
     * must provide the method @c operator()(std::vector<T>&, idx_t), where T =
     * tlapack::type_t<array_t> and idx_t = tlapack::size_type<array_t>. The
     * output of the functor satisfies the concept tlapack::concepts::Vector.
     *
     * - @c tlapack::traits::CreateStaticFunctor<tlapack::matrix_type<array_t>,
     * m, n, int> must provide the method @c operator()(T*), where T =
     * tlapack::type_t<array_t>. The output of the functor satisfies the concept
     * tlapack::concepts::Matrix.
     *
     * - @c tlapack::traits::CreateStaticFunctor<tlapack::vector_type<array_t>,
     * n, int> must provide the method @c operator()(T*), where T =
     * tlapack::type_t<array_t>. The output of the functor satisfies the concept
     * tlapack::concepts::Vector.
     *
     * @tparam array_t Matrix or vector type.
     *
     * @ingroup concepts
     */
    template <typename array_t>
    concept ConstructableArray = Matrix<matrix_type<array_t>>&&
        Vector<vector_type<array_t>>&& requires(std::vector<type_t<array_t>>& v,
                                                type_t<array_t>* ptr)
    {
        {
            Create<matrix_type<array_t>>()(v, 2, 3)
        }
        ->Matrix<>;
        {
            Create<vector_type<array_t>>()(v, 2)
        }
        ->Vector<>;
        {
            CreateStatic<matrix_type<array_t>, 5, 6>()(ptr)
        }
        ->Matrix<>;
        {
            CreateStatic<vector_type<array_t>, 5>()(ptr)
        }
        ->Vector<>;
    };

    /** @interface tlapack::concepts::ConstructableMatrix
     * @brief Concept for matrices that implement
     * tlapack::traits::CreateFunctor and tlapack::traits::CreateStaticFunctor.
     *
     * A constructable matrix must provide an implementation of the functors
     * tlapack::traits::CreateFunctor and tlapack::traits::CreateStaticFunctor
     * for the type @c matrix_t. The outputs of the functors satisfy the concept
     * tlapack::concepts::Matrix.
     *
     * - @c tlapack::traits::CreateFunctor<matrix_t, int> must provide the
     * method @c operator()(std::vector<T>&, idx_t, idx_t), where T =
     * tlapack::type_t<matrix_t> and idx_t = tlapack::size_type<matrix_t>.
     *
     * - @c tlapack::traits::CreateStaticFunctor<matrix_t, m, n, int> must
     * provide the method @c operator()(T*), where T =
     * tlapack::type_t<matrix_t>.
     *
     * Moreover, a constructable matrix must also satisfy the concept
     * tlapack::concepts::ConstructableArray.
     *
     * @tparam matrix_t Matrix type.
     *
     * @ingroup concepts
     */
    template <typename matrix_t>
    concept ConstructableMatrix =
        Matrix<matrix_t>&& ConstructableArray<matrix_t>&& requires(
            std::vector<type_t<matrix_t>>& v, type_t<matrix_t>* ptr)
    {
        {
            Create<matrix_t>()(v, 2, 3)
        }
        ->Matrix<>;
        {
            CreateStatic<matrix_t, 5, 6>()(ptr)
        }
        ->Matrix<>;
    };

    /** @interface tlapack::concepts::ConstructableAndSliceableMatrix
     * @brief Concept for matrices that satisfy
     * tlapack::concepts::ConstructableMatrix and
     * tlapack::concepts::SliceableMatrix.
     *
     * @tparam matrix_t Matrix type.
     *
     * @ingroup concepts
     */
    template <typename matrix_t>
    concept ConstructableAndSliceableMatrix =
        SliceableMatrix<matrix_t>&& ConstructableMatrix<matrix_t>;

    /** @interface tlapack::concepts::ConstructableVector
     * @brief Concept for vectors that implement tlapack::traits::CreateFunctor
     * and tlapack::traits::CreateStaticFunctor.
     *
     * A constructable vector must provide an implementation of the functors
     * tlapack::traits::CreateFunctor and tlapack::traits::CreateStaticFunctor
     * for the type @c vector_t. The output of the functor satisfies the concept
     * tlapack::concepts::Vector.
     *
     * - @c tlapack::traits::CreateFunctor<vector_t, int> must provide the
     * method @c operator()(std::vector<T>&, idx_t), where T =
     * tlapack::type_t<vector_t> and idx_t = tlapack::size_type<vector_t>.
     *
     * - @c tlapack::traits::CreateStaticFunctor<vector_t, n, int> must provide
     * the method @c operator()(T*), where T = tlapack::type_t<vector_t>.
     *
     * Moreover, a constructable vector must also satisfy the concept
     * tlapack::concepts::ConstructableArray.
     *
     * @tparam vector_t Vector type.
     *
     * @ingroup concepts
     */
    template <typename vector_t>
    concept ConstructableVector =
        Vector<vector_t>&& ConstructableArray<vector_t>&& requires(
            std::vector<type_t<vector_t>>& v, type_t<vector_t>* ptr)
    {
        {
            Create<vector_t>()(v, 2)
        }
        ->Vector<>;
        {
            CreateStatic<vector_t, 5>()(ptr)
        }
        ->Vector<>;
    };

    /** @interface tlapack::concepts::ConstructableAndSliceableVector
     * @brief Concept for vectors that satisfy
     * tlapack::concepts::ConstructableVector and
     * tlapack::concepts::SliceableVector.
     *
     * @tparam vector_t Vector type.
     *
     * @ingroup concepts
     */
    template <typename vector_t>
    concept ConstructableAndSliceableVector =
        SliceableVector<vector_t>&& ConstructableVector<vector_t>;

}  // namespace concepts
}  // namespace tlapack

    /// Macro for tlapack::concepts::Matrix compatible with C++17.
    #define TLAPACK_MATRIX tlapack::concepts::Matrix

    /// Macro for tlapack::concepts::SliceableMatrix compatible with C++17.
    #define TLAPACK_SMATRIX tlapack::concepts::SliceableMatrix

    /// Macro for tlapack::concepts::SliceableTransposeMatrix compatible with
    /// C++17.
    #define TLAPACK_STMATRIX tlapack::concepts::SliceableTransposeMatrix

    /// Macro for tlapack::concepts::Vector compatible with C++17.
    #define TLAPACK_VECTOR tlapack::concepts::Vector

    /// Macro for tlapack::concepts::SliceableVector compatible with C++17.
    #define TLAPACK_SVECTOR tlapack::concepts::SliceableVector

    /// Macro for tlapack::concepts::Workspace compatible with C++17.
    #define TLAPACK_WORKSPACE tlapack::concepts::Workspace

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

    /// Macro for tlapack::concepts::LegacyArray compatible with C++17.
    #define TLAPACK_LEGACY_ARRAY tlapack::concepts::LegacyArray

    /// Macro for tlapack::concepts::LegacyMatrix compatible with C++17.
    #define TLAPACK_LEGACY_MATRIX tlapack::concepts::LegacyMatrix

    /// Macro for tlapack::concepts::LegacyVector compatible with C++17.
    #define TLAPACK_LEGACY_VECTOR tlapack::concepts::LegacyVector

    /// Macro for tlapack::concepts::ConstructableMatrix compatible with C++17.
    #define TLAPACK_CMATRIX tlapack::concepts::ConstructableMatrix

    /// Macro for tlapack::concepts::ConstructableAndSliceableMatrix compatible
    /// with C++17.
    #define TLAPACK_CSMATRIX tlapack::concepts::ConstructableAndSliceableMatrix

    /// Macro for tlapack::concepts::ConstructableVector compatible with C++17.
    #define TLAPACK_CVECTOR tlapack::concepts::ConstructableVector

    /// Macro for tlapack::concepts::ConstructableAndSliceableVector compatible
    /// with C++17.
    #define TLAPACK_CSVECTOR tlapack::concepts::ConstructableAndSliceableVector
#else
    // Concepts are a C++20 feature, so just define them as `class` for earlier
    // versions
    #define TLAPACK_MATRIX class
    #define TLAPACK_SMATRIX class
    #define TLAPACK_STMATRIX class

    #define TLAPACK_VECTOR class
    #define TLAPACK_SVECTOR class

    #define TLAPACK_WORKSPACE class

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

    #define TLAPACK_LEGACY_ARRAY class
    #define TLAPACK_LEGACY_MATRIX class
    #define TLAPACK_LEGACY_VECTOR class

    #define TLAPACK_CMATRIX class
    #define TLAPACK_CSMATRIX class
    #define TLAPACK_CVECTOR class
    #define TLAPACK_CSVECTOR class
#endif

#endif  // TLAPACK_ARRAY_CONCEPTS_HH
