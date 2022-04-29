/// @file debug_utils.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_DEBUG_UTILS_HH__
#define __TLAPACK_DEBUG_UTILS_HH__

#include <iostream>
#include <iomanip>
#include <sstream>
#include <complex>

#include "legacy_api/legacyArray.hpp"
#include "blas/types.hpp"

namespace lapack
{

    /**
     * @brief Prints a matrix a to std::out
     *
     * @param A m by n matrix
     */
    template <typename matrix_t>
    void print_matrix(const matrix_t &A)
    {
        using idx_t = blas::size_type<matrix_t>;
        const idx_t m = blas::nrows(A);
        const idx_t n = blas::ncols(A);

        for (idx_t i = 0; i < m; ++i)
        {
            std::cout << std::endl;
            for (idx_t j = 0; j < n; ++j)
                std::cout << std::setw(16) << A(i, j) << " ";
        }
    }

    //
    // GDB doesn't handle templates well, so we explicitly define some versions of the functions
    // for common template arguments
    //
    void print_matrix_r(const blas::legacyMatrix<float, blas::Layout::ColMajor> &A)
    {
        print_matrix(A);
    }
    void print_matrix_d(const blas::legacyMatrix<double, blas::Layout::ColMajor> &A)
    {
        print_matrix(A);
    }
    void print_matrix_c(const blas::legacyMatrix<std::complex<float>, blas::Layout::ColMajor> &A)
    {
        print_matrix(A);
    }
    void print_matrix_z(const blas::legacyMatrix<std::complex<double>, blas::Layout::ColMajor> &A)
    {
        print_matrix(A);
    }

    /**
     * @brief Constructs a json string representing the matrix
     *        for use with vscode-debug-visualizer
     *
     * @return String containing JSON representation of A
     *
     * @param A m by n matrix
     */
    template <typename matrix_t>
    std::string visualize_matrix(const matrix_t &A)
    {
        using idx_t = blas::size_type<matrix_t>;
        const idx_t m = blas::nrows(A);
        const idx_t n = blas::ncols(A);

        std::stringstream stream;
        stream << "{ \"kind\":{ \"plotly\": true },\"data\":[{";

        // Add the header (matrix index)
        stream << "\"header\":{\"values\":[";
        for (idx_t i = 0; i < m; ++i)
        {
            stream << i;
            if (i + 1 < m)
                stream << ", ";
        }
        stream << "]},";

        // Add the matrix values
        stream << "\"cells\":{\"values\":[";
        for (idx_t i = 0; i < m; ++i)
        {
            stream << "[";
            for (idx_t j = 0; j < n; ++j)
            {
                stream << std::setprecision(3) << A(i, j);
                if (j + 1 < n)
                    stream << ", ";
            }
            stream << "]";
            if (i + 1 < m)
                stream << ", ";
        }

        stream << "]},";
        stream << "\"type\": \"table\"}],\"layout\": {}}";

        return stream.str();
    }

    //
    // GDB doesn't handle templates well, so we explicitly define some versions of the functions
    // for common template arguments
    //
    std::string visualize_matrix_r(const blas::legacyMatrix<float, blas::Layout::ColMajor> &A)
    {
        return visualize_matrix(A);
    }
    std::string visualize_matrix_d(const blas::legacyMatrix<double, blas::Layout::ColMajor> &A)
    {
        return visualize_matrix(A);
    }
    std::string visualize_matrix_c(const blas::legacyMatrix<std::complex<float>, blas::Layout::ColMajor> &A)
    {
        return visualize_matrix(A);
    }
    std::string visualize_matrix_z(const blas::legacyMatrix<std::complex<double>, blas::Layout::ColMajor> &A)
    {
        return visualize_matrix(A);
    }
}

#endif // __TLAPACK_DEBUG_UTILS_HH