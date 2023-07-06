/// @file arrayTraits.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ARRAY_TRAITS_HH
#define TLAPACK_ARRAY_TRAITS_HH

#include "tlapack/base/types.hpp"

namespace tlapack {

using std::enable_if_t;
using std::is_same_v;

namespace traits {

    /**
     * @brief Entry type trait.
     *
     * The entry type is defined on @c type_trait<array_t>::type.
     *
     * @tparam T A matrix or vector class
     */
    template <class T, typename = int>
    struct entry_type_trait {
        using type = void;
    };

    /**
     * @brief Size type trait.
     *
     * The size type is defined on @c sizet_trait<array_t>::type.
     *
     * @tparam T A matrix or vector class
     */
    template <class T, typename = int>
    struct size_type_trait {
        using type = std::size_t;
    };

    /**
     * @brief Trait to determine the layout of a given data structure.
     *
     * This is used to verify if optimized routines can be used.
     *
     * @tparam array_t Data structure.
     * @tparam class If this is not an int, then the trait is not defined.
     *
     * @ingroup abstract_matrix
     */
    template <class array_t, class = int>
    struct layout_trait {
        static constexpr Layout value = Layout::Unspecified;
    };

    /**
     * @brief Trait to determine the transpose of matrices from class matrix_t.
     *
     * @tparam matrix_t Data structure.
     * @tparam class If this is not an int, then the trait is not defined.
     *
     * @ingroup abstract_matrix
     */
    // template <class matrix_t, class = int>
    // struct transpose_type_trait;

    /**
     * @brief Functor for data creation
     *
     * This is a boilerplate. It must be specialized for each class.
     *
     * @ingroup abstract_matrix
     */
    template <class matrix_t, class = int>
    struct CreateFunctor {
        static_assert(false && sizeof(matrix_t),
                      "Must use correct specialization");

        /**
         * @brief Creates an object (matrix or vector)
         *
         * @param[out] v
         *      Vector that can be used to reference allocated memory.
         * @param[in] m Number of rows
         * @param[in] n Number of Columns
         *
         * @return The new object
         */
        template <class T, class idx_t>
        inline constexpr auto operator()(std::vector<T>& v,
                                         idx_t m,
                                         idx_t n = 1) const
        {
            return matrix_t();
        }

        /**
         * @brief Creates a matrix
         *
         * @param[in] W
         *      Workspace that references to allocated memory.
         * @param[in] m Number of rows
         * @param[in] n Number of Columns
         * @param[out] rW
         *      On exit, receives the updated workspace, i.e., that references
         *      remaining allocated memory.
         *
         * @return The new object
         */
        template <class idx_t, class Workspace>
        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         idx_t n,
                                         Workspace& rW) const
        {
            return matrix_t();
        }

        /**
         * @brief Creates a vector
         *
         * @param[in] W
         *      Workspace that references to allocated memory.
         * @param[in] m Size of the vector
         * @param[out] rW
         *      On exit, receives the updated workspace, i.e., that references
         *      remaining allocated memory.
         *
         * @return The new object
         */
        template <class idx_t, class Workspace>
        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         Workspace& rW) const
        {
            return matrix_t();
        }

        /**
         * @brief Creates an object (matrix or vector)
         *
         * @param[in] W
         *      Vector that can be used to reference allocated memory.
         * @param[in] m Number of rows
         * @param[in] n Number of Columns
         *
         * @return The new object
         */
        template <class idx_t, class Workspace>
        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         idx_t n = 1) const
        {
            return matrix_t();
        }
    };

    // Matrix and vector type deduction:

    /**
     * @brief Matrix type deduction
     *
     * The deduction for two types must be implemented elsewhere. For example,
     * the plugins for LegacyArray, Eigen and mdspan all have their own
     * implementation.
     *
     * @tparam matrix_t List of matrix types. The last type must be an int.
     *
     * @ingroup abstract_matrix
     */
    template <typename... matrix_t>
    struct matrix_type_traits;

    /// Matrix type deduction for three or more types
    template <typename matrixA_t, typename matrixB_t, typename... matrix_t>
    struct matrix_type_traits<matrixA_t, matrixB_t, matrix_t...> {
        using type = typename matrix_type_traits<
            typename matrix_type_traits<matrixA_t, matrixB_t, int>::type,
            matrix_t...>::type;
    };

    /**
     * @brief Vector type deduction
     *
     * The deduction for two types must be implemented elsewhere. For example,
     * the plugins for LegacyArray, Eigen and mdspan all have their own
     * implementation.
     *
     * @tparam vector_t List of vector types. The last type must be an int.
     *
     * @ingroup abstract_matrix
     */
    template <typename... vector_t>
    struct vector_type_traits;

    /// Vector type deduction for one type
    template <typename vector_t>
    struct vector_type_traits<vector_t, int> {
        using type = typename vector_type_traits<vector_t, vector_t, int>::type;
    };

    /// Vector type deduction for three or more types
    template <typename vectorA_t, typename vectorB_t, typename... vector_t>
    struct vector_type_traits<vectorA_t, vectorB_t, vector_t...> {
        using type = typename vector_type_traits<
            typename vector_type_traits<vectorA_t, vectorB_t, int>::type,
            vector_t...>::type;
    };
}  // namespace traits

/// @brief Alias for @c traits::layout_trait<,int>::value.
/// @ingroup abstract_matrix
template <class array_t>
constexpr Layout layout = traits::layout_trait<array_t, int>::value;

/**
 * @brief Alias for @c traits::CreateFunctor<,int>.
 *
 * Usage:
 * @code{.cpp}
 * // matrix_t is a predefined type at this point
 *
 * using T = tlapack::type_t<matrix_t>;
 * using idx_t = tlapack::size_type<matrix_t>;
 *
 * tlapack::Create<matrix_t> new_matrix; // Creates the functor
 *
 * idx_t m = 11;
 * idx_t n = 6;
 *
 * std::vector<T> A_container; // Empty vector
 * auto A = new_matrix(A_container, m, n); // Initialize A_container if needed
 *
 * tlapack::VectorOfBytes B_container; // Empty vector
 * auto B = new_matrix(
 *  tlapack::alloc_workspace( B_container, m*n*sizeof(T) ),
 *  m, n ); // B_container stores allocated memory
 *
 * tlapack::VectorOfBytes C_container; // Empty vector
 * Workspace W; // Empty workspace
 * auto C = new_matrix(
 *  tlapack::alloc_workspace( C_container, m*n*sizeof(T) ),
 *  m, n, W ); // W receives the updated workspace, i.e., without the space
 * taken by C
 * @endcode
 *
 * @ingroup abstract_matrix
 */
template <class T>
using Create = traits::CreateFunctor<T, int>;

/// Alias for @c traits::transpose_type_trait<,int>::type.
/// @ingroup abstract_matrix
template <class T>
using transpose_type =
    typename traits::matrix_type_traits<T, int>::transpose_type;

/// Alias for @c traits::matrix_type<,int>::type.
/// @ingroup abstract_matrix
template <typename... matrix_t>
using matrix_type = typename traits::matrix_type_traits<matrix_t..., int>::type;

/// Alias for @c traits::vector_type<,int>::type.
/// @ingroup abstract_matrix
template <typename... vector_t>
using vector_type = typename traits::vector_type_traits<vector_t..., int>::type;

/// Entry type of a matrix or vector.
template <class T>
using type_t = typename traits::entry_type_trait<T, int>::type;

/// Size type of a matrix or vector.
template <class T>
using size_type = typename traits::size_type_trait<T, int>::type;

}  // namespace tlapack

#endif  // TLAPACK_ARRAY_TRAITS_HH
