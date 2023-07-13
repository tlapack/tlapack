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
#include "tlapack/base/workspace.hpp"

namespace tlapack {

using std::enable_if_t;
using std::is_same_v;

namespace traits {

    /**
     * @brief Entry type trait.
     *
     * The entry type is defined on @c type_trait<array_t,int>::type.
     *
     * @note This trait does not need to be specialized for each class as
     * @c utils.hpp already provides a default implementation for matrices and
     * vectors based on the entry access operator.
     *
     * @see tlapack::concepts::Matrix
     * @see tlapack::concepts::Vector
     *
     * @tparam T A matrix or vector class
     * @tparam class If this is not an int, then the trait is not defined.
     */
    template <class T, class = int>
    struct entry_type_trait {
        using type = void;
    };

    /**
     * @brief Size type trait.
     *
     * The size type is defined on @c sizet_trait<array_t,int>::type.
     *
     * @note This trait does not need to be specialized for each class as
     * @c utils.hpp already provides a default implementation for matrices and
     * vectors based on the functions @c nrows(T) and @c size(T).
     *
     * @see tlapack::concepts::Matrix
     * @see tlapack::concepts::Vector
     *
     * @tparam T A matrix or vector class
     * @tparam class If this is not an int, then the trait is not defined.
     */
    template <class T, class = int>
    struct size_type_trait {
        using type = std::size_t;
    };

    /**
     * @brief Trait to determine the layout of a given data structure.
     *
     * The layout is defined on @c layout_trait<array_t,int>::value.
     *
     * This is used to verify if optimized implementations can be used.
     *
     * @tparam array_t Data structure.
     * @tparam class If this is not an int, then the trait is not defined.
     */
    template <class array_t, class = int>
    struct layout_trait {
        static constexpr Layout value = Layout::Unspecified;
    };

    /**
     * @brief Functor for data creation
     *
     * This is a boilerplate. It must be specialized for each class.
     * See tlapack::Create for examples of usage.
     *
     * @tparam matrix_t Data structure.
     * @tparam class If this is not an int, then the trait is not defined.
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
         * @brief Creates a column-major matrix from a given workspace
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
        template <class idx_t>
        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         idx_t n,
                                         Workspace& rW) const
        {
            return matrix_t();
        }

        /**
         * @brief Creates a column-major matrix from a given workspace
         *
         * @param[in] W
         *      Workspace that references to allocated memory.
         * @param[in] m Number of rows
         * @param[in] n Number of Columns
         *
         * @return The new object
         */
        template <class idx_t>
        inline constexpr auto operator()(const Workspace& W,
                                         idx_t m,
                                         idx_t n) const
        {
            return matrix_t();
        }

        /**
         * @brief Creates a strided vector from a given workspace
         *
         * @param[in] W
         *      Workspace that references to allocated memory.
         * @param[in] n Size of the vector
         * @param[out] rW
         *      On exit, receives the updated workspace, i.e., that references
         *      remaining allocated memory.
         *
         * @return The new object
         */
        template <class idx_t>
        inline constexpr auto operator()(const Workspace& W,
                                         idx_t n,
                                         Workspace& rW) const
        {
            return matrix_t();
        }

        /**
         * @brief Creates a strided vector from a given workspace
         *
         * @param[in] W
         *      Workspace that references to allocated memory.
         * @param[in] n Size of the vector
         *
         * @return The new object
         */
        template <class idx_t>
        inline constexpr auto operator()(const Workspace& W, idx_t n) const
        {
            return matrix_t();
        }
    };

    // Matrix and vector type deduction:

    /**
     * @brief Matrix type deduction
     *
     * The deduction for one and two types must be implemented elsewhere. For
     * example, the plugins for Eigen and mdspan matrices have their own
     * implementation.
     *
     * @tparam matrix_t List of matrix types. The last type must be an int.
     */
    template <class... matrix_t>
    struct matrix_type_traits;

    // Matrix type deduction for one type
    template <class matrix_t>
    struct matrix_type_traits<matrix_t, int> {
        using type = typename matrix_type_traits<matrix_t, matrix_t, int>::type;
    };

    // Matrix type deduction for three or more types
    template <class matrixA_t, class matrixB_t, class... matrix_t>
    struct matrix_type_traits<matrixA_t, matrixB_t, matrix_t...> {
        using type = typename matrix_type_traits<
            typename matrix_type_traits<matrixA_t, matrixB_t, int>::type,
            matrix_t...>::type;
    };

    /**
     * @brief Vector type deduction
     *
     * The deduction for two types must be implemented elsewhere. For example,
     * the plugins for Eigen and mdspan matrices have their own implementation.
     *
     * The deduced type is defined on @c vector_type_traits<vector_t...>::type.
     *
     * @tparam vector_t List of vector types. The last type must be an int.
     */
    template <class... vector_t>
    struct vector_type_traits;

    // Vector type deduction for one type
    template <class vector_t>
    struct vector_type_traits<vector_t, int> {
        using type = typename vector_type_traits<vector_t, vector_t, int>::type;
    };

    // Vector type deduction for three or more types
    template <class vectorA_t, class vectorB_t, class... vector_t>
    struct vector_type_traits<vectorA_t, vectorB_t, vector_t...> {
        using type = typename vector_type_traits<
            typename vector_type_traits<vectorA_t, vectorB_t, int>::type,
            vector_t...>::type;
    };
}  // namespace traits

// Aliases for the traits:

/// Entry type of a matrix or vector.
template <class T>
using type_t = typename traits::entry_type_trait<T, int>::type;

/// Size type of a matrix or vector.
template <class T>
using size_type = typename traits::size_type_trait<T, int>::type;

/// Layout of a matrix or vector.
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
 */
template <class T>
using Create = traits::CreateFunctor<T, int>;

/// Common matrix type deduced from the list of types.
template <class... matrix_t>
using matrix_type = typename traits::matrix_type_traits<matrix_t..., int>::type;

/// Common vector type deduced from the list of types.
template <class... vector_t>
using vector_type = typename traits::vector_type_traits<vector_t..., int>::type;

}  // namespace tlapack

#endif  // TLAPACK_ARRAY_TRAITS_HH
