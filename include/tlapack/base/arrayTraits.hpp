/// @file arrayTraits.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ARRAY_TRAITS_HH
#define TLAPACK_ARRAY_TRAITS_HH

#include "tlapack/base/types.hpp"

namespace tlapack {

// C++ standard utils:
using std::enable_if_t;
using std::is_same_v;

namespace traits {

    /**
     * @brief Entry type trait.
     *
     * The entry type is defined on @c type_trait<array_t,int>::type. Use the
     * tlapack::type_t alias instead.
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
     * The size type is defined on @c sizet_trait<array_t,int>::type. Use the
     * tlapack::size_type alias instead.
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
     * The layout is defined on @c layout_trait<array_t,int>::value. Use the
     * tlapack::layout alias instead.
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
         * @brief Creates a m-by-n matrix with entries of type T
         *
         * @param[in,out] v
         *          On entry, empty vector with size 0.
         *          On exit, vector that may contain allocated memory.
         * @param[in] m Number of rows of the new matrix
         * @param[in] n Number of columns of the new matrix
         *
         * @return The new m-by-n matrix
         */
        template <class T, class idx_t>
        constexpr auto operator()(std::vector<T>& v, idx_t m, idx_t n = 1) const
        {
            return matrix_t();
        }

        /**
         * @brief Creates a vector of size n with entries of type T
         *
         * @param[out] v
         *          On entry, empty vector with size 0.
         *          On exit, vector that may contain allocated memory.
         * @param[in] n Size of the new vector
         *
         * @return The new vector of size n
         */
        template <class T, class idx_t>
        constexpr auto operator()(std::vector<T>& v, idx_t n) const
        {
            return matrix_t();
        }
    };

    /**
     * @brief Functor for data creation with static size
     *
     * This is a boilerplate. It must be specialized for each class.
     * See tlapack::Create for examples of usage.
     *
     * @tparam matrix_t Data structure.
     * @tparam m Number of rows of the new matrix or number of elements of the
     * new vector. m >= 0.
     * @tparam n Number of columns of the new matrix. If n == -1 then the
     * functor creates a vector of size @c m. Otherwise, n >= 0.
     * @tparam class If this is not an int, then the trait is not defined.
     */
    template <class matrix_t, int m, int n, class = int>
    struct CreateStaticFunctor {
        static_assert(false && sizeof(matrix_t),
                      "Must use correct specialization");
        static_assert(m >= 0 && n >= -1);

        /**
         * @brief Creates a m-by-n matrix or, if n == -1, a vector of size m
         *
         * @tparam T Entry type.
         *
         * @param[in] v Pointer to the memory that may be used to store the
         * matrix or vector.
         *
         * @return The new m-by-n matrix or vector of size m
         */
        template <typename T>
        constexpr auto operator()(T* v) const
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
     * implementation. The deduction for three or more types is implemented
     * here. Use tlapack::matrix_type alias to get the deduced type.
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
     * The deduction for three or more types is implemented here. Use
     * tlapack::vector_type alias to get the deduced type.
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
 * tlapack::Create<matrix_t> new_matrix; // Creates the functor
 *
 * size_t m = 11;
 * size_t n = 6;
 *
 * std::vector<float> A_container; // Empty vector
 * auto A = new_matrix(A_container, m, n); // Initialize A_container if needed
 * @endcode
 */
template <class T>
using Create = traits::CreateFunctor<T, int>;

/**
 * @brief Alias for @c traits::CreateStaticFunctor<,int>.
 *
 * Usage:
 * @code{.cpp}
 * // matrix_t and vector_t are predefined types at this point
 *
 * constexpr size_t M = 8;
 * constexpr size_t N = 6;
 * tlapack::CreateStatic<matrix_t, M, N> new_MbyN_matrix; // Creates the functor
 * tlapack::CreateStatic<vector_t, N> new_N_vector; // Creates the functor
 *
 * float A_container[M * N]; // Array of size M * N
 * auto A = new_MbyN_matrix(A_container);
 *
 * double B_container[N]; // Array of size N
 * auto B = new_N_vector(B_container);
 * @endcode
 */
template <class T, int m, int n = -1>
using CreateStatic = traits::CreateStaticFunctor<T, m, n, int>;

/// Common matrix type deduced from the list of types.
template <class... matrix_t>
using matrix_type = typename traits::matrix_type_traits<matrix_t..., int>::type;

/// Common vector type deduced from the list of types.
template <class... vector_t>
using vector_type = typename traits::vector_type_traits<vector_t..., int>::type;

}  // namespace tlapack

#endif  // TLAPACK_ARRAY_TRAITS_HH
