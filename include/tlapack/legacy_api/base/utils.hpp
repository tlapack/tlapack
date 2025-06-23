/// @file legacy_api/base/utils.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_UTILS_HH
#define TLAPACK_LEGACY_UTILS_HH

#include "tlapack/legacy_api/base/legacyArray.hpp"

/**
 * @brief Creates two vector objects and executes an expression with them.
 *
 * @param[in] x Name of the first vector to be used in the expression.
 * @param[in] TX Type of the elements of the first vector.
 * @param[in] n Number of elements in the first vector.
 * @param[in] X Pointer to the first element of the first vector.
 * @param[in] incx Stride between elements of the first vector.
 * @param[in] y Name of the second vector to be used in the expression.
 * @param[in] TY Type of the elements of the second vector.
 * @param[in] m Number of elements in the second vector.
 * @param[in] Y Pointer to the first element of the second vector.
 * @param[in] incy Stride between elements of the second vector.
 * @param[in] expr Expression to be executed.
 *
 * @ingroup legacy_api
 *
 */
#define tlapack_expr_with_2vectors(x, TX, n, X, incx, y, TY, m, Y, incy, expr) \
    do {                                                                       \
        using tlapack::legacy::internal::create_vector;                        \
        using tlapack::legacy::internal::create_backward_vector;               \
        if (incx == 1) {                                                       \
            auto x = create_vector((TX*)X, n);                                 \
            tlapack_expr_with_vector(y, TY, m, Y, incy, expr);                 \
        }                                                                      \
        else if (incx == -1) {                                                 \
            auto x = create_backward_vector((TX*)X, n);                        \
            tlapack_expr_with_vector(y, TY, m, Y, incy, expr);                 \
        }                                                                      \
        else if (incx > 1) {                                                   \
            auto x = create_vector((TX*)X, n, incx);                           \
            tlapack_expr_with_vector(y, TY, m, Y, incy, expr);                 \
        }                                                                      \
        else {                                                                 \
            auto x = create_backward_vector((TX*)X, n, -incx);                 \
            tlapack_expr_with_vector(y, TY, m, Y, incy, expr);                 \
        }                                                                      \
    } while (false)

/**
 * @brief Creates a vector object and executes an expression with it.
 *
 * @param[in] x Name of the vector to be used in the expression.
 * @param[in] TX Type of the elements of the vector.
 * @param[in] n Number of elements in the vector.
 * @param[in] X Pointer to the first element of the vector.
 * @param[in] incx Stride between elements of the vector.
 * @param[in] expr Expression to be executed with the vector.
 *
 * @ingroup legacy_api
 *
 */
#define tlapack_expr_with_vector(x, TX, n, X, incx, expr)        \
    do {                                                         \
        using tlapack::legacy::internal::create_vector;          \
        using tlapack::legacy::internal::create_backward_vector; \
        if (incx == 1) {                                         \
            auto x = create_vector((TX*)X, n);                   \
            expr;                                                \
        }                                                        \
        else if (incx == -1) {                                   \
            auto x = create_backward_vector((TX*)X, n);          \
            expr;                                                \
        }                                                        \
        else if (incx > 1) {                                     \
            auto x = create_vector((TX*)X, n, incx);             \
            expr;                                                \
        }                                                        \
        else {                                                   \
            auto x = create_backward_vector((TX*)X, n, -incx);   \
            expr;                                                \
        }                                                        \
    } while (false)

/**
 * @brief Creates a vector object and executes an expression with it.
 *
 * This macro is used when the stride between elements of the vector is
 * non-negative.
 *
 * @param[in] x Name of the vector to be used in the expression.
 * @param[in] TX Type of the elements of the vector.
 * @param[in] n Number of elements in the vector.
 * @param[in] X Pointer to the first element of the vector.
 * @param[in] incx Stride between elements of the vector.
 * @param[in] expr Expression to be executed with the vector.
 *
 * @ingroup legacy_api
 *
 */
#define tlapack_expr_with_vector_positiveInc(x, TX, n, X, incx, expr) \
    do {                                                              \
        using tlapack::legacy::internal::create_vector;               \
        if (incx == 1) {                                              \
            auto x = create_vector((TX*)X, n);                        \
            expr;                                                     \
        }                                                             \
        else {                                                        \
            auto x = create_vector((TX*)X, n, incx);                  \
            expr;                                                     \
        }                                                             \
    } while (false)

#endif  // TLAPACK_LEGACY_UTILS_HH
