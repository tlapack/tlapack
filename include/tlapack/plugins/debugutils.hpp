/// @file debugutils.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_DEBUG_UTILS_HH
#define TLAPACK_DEBUG_UTILS_HH

#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * @brief Prints a matrix a to std::out
 *
 * @param A m by n matrix
 */
template <AbstractMatrix matrix_t>
void print_matrix(const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << std::setw(16) << A(i, j) << " ";
    }
}

/**
 * @brief Constructs a json string representing the matrix
 *        for use with vscode-debug-visualizer
 *
 * @return String containing JSON representation of A
 *
 * @param A m by n matrix
 */
template <AbstractMatrix matrix_t>
std::string visualize_matrix_text(const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    const int width = is_complex<type_t<matrix_t>>::value ? 25 : 10;

    std::stringstream stream;
    stream << "{ \"kind\":{ \"text\": true },\"text\": \"";

    // Col indices
    for (idx_t j = 0; j < n; ++j)
        stream << std::setw(width) << j << " ";
    stream << "\\n";

    // Add the matrix values
    for (idx_t i = 0; i < m; ++i) {
        for (idx_t j = 0; j < n; ++j)
            stream << std::setw(width) << std::setprecision(3) << A(i, j)
                   << " ";
        stream << "\\n";
    }

    stream << "\"}";

    return stream.str();
}

/**
 * @brief Constructs a json string representing the matrix
 *        for use with vscode-debug-visualizer
 *
 * @return String containing JSON representation of A
 *
 * @param A m by n matrix
 */
template <AbstractMatrix matrix_t>
std::string visualize_matrix_table(const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    std::stringstream stream;
    stream << "{ \"kind\":{ \"plotly\": true },\"data\":[{";

    // Add the header (matrix index)
    stream << "\"header\":{\"values\":[";
    for (idx_t j = 0; j < n; ++j) {
        stream << j;
        if (j + 1 < n) stream << ", ";
    }
    stream << "]},";

    // Add the matrix values
    stream << "\"cells\":{\"values\":[";
    for (idx_t j = 0; j < n; ++j) {
        stream << "[";
        for (idx_t i = 0; i < m; ++i) {
            stream << "\"" << std::setprecision(3) << A(i, j) << "\"";
            if (i + 1 < m) stream << ", ";
        }
        stream << "]";
        if (j + 1 < n) stream << ", ";
    }

    stream << "]},";
    stream << "\"type\": \"table\"}],\"layout\": {}}";

    return stream.str();
}

/**
 * @brief Constructs a json string representing the matrix
 *        for use with vscode-debug-visualizer
 *
 * @return String containing JSON representation of A
 *
 * @param A m by n matrix
 */
template <AbstractMatrix matrix_t>
std::string visualize_matrix(const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;
    const idx_t n = std::max(ncols(A), nrows(A));

    if (n > 7)
        return visualize_matrix_text(A);
    else
        return visualize_matrix_table(A);
}

}  // namespace tlapack

#endif  // TLAPACK_DEBUG_UTILS_HH
