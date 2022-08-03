/// @file tlapack_debugutils.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_DEBUG_UTILS_HH
#define TLAPACK_DEBUG_UTILS_HH

#include <iostream>
#include <iomanip>
#include <sstream>
#include <complex>
#include <fstream>

#include "tlapack/legacy_api/legacyArray.hpp"
#include "tlapack/base/types.hpp"
#include "tlapack/base/utils.hpp"

namespace tlapack
{

    /**
     * @brief Prints a matrix a to std::out
     *
     * @param A m by n matrix
     */
    template <typename matrix_t>
    void print_matrix(const matrix_t &A)
    {
        using idx_t = size_type<matrix_t>;
        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

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
    void print_matrix_r(const legacyMatrix<float, Layout::ColMajor> &A)
    {
        print_matrix(A);
    }
    void print_matrix_d(const legacyMatrix<double, Layout::ColMajor> &A)
    {
        print_matrix(A);
    }
    void print_matrix_c(const legacyMatrix<std::complex<float>, Layout::ColMajor> &A)
    {
        print_matrix(A);
    }
    void print_matrix_z(const legacyMatrix<std::complex<double>, Layout::ColMajor> &A)
    {
        print_matrix(A);
    }
    void print_rowmajormatrix_r(const legacyMatrix<float, Layout::RowMajor> &A)
    {
        print_matrix(A);
    }
    void print_rowmajormatrix_d(const legacyMatrix<double, Layout::RowMajor> &A)
    {
        print_matrix(A);
    }
    void print_rowmajormatrix_c(const legacyMatrix<std::complex<float>, Layout::RowMajor> &A)
    {
        print_matrix(A);
    }
    void print_rowmajormatrix_z(const legacyMatrix<std::complex<double>, Layout::RowMajor> &A)
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
    std::string visualize_matrix_text(const matrix_t &A)
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
        for (idx_t i = 0; i < m; ++i)
        {
            for (idx_t j = 0; j < n; ++j)
                stream << std::setw(width) << std::setprecision(3) << A(i, j) << " ";
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
    template <typename matrix_t>
    std::string visualize_matrix_table(const matrix_t &A)
    {
        using idx_t = size_type<matrix_t>;
        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        std::stringstream stream;
        stream << "{ \"kind\":{ \"plotly\": true },\"data\":[{";

        // Add the header (matrix index)
        stream << "\"header\":{\"values\":[";
        for (idx_t j = 0; j < n; ++j)
        {
            stream << j;
            if (j + 1 < n)
                stream << ", ";
        }
        stream << "]},";

        // Add the matrix values
        stream << "\"cells\":{\"values\":[";
        for (idx_t j = 0; j < n; ++j)
        {
            stream << "[";
            for (idx_t i = 0; i < m; ++i)
            {
                stream << "\"" << std::setprecision(3) << A(i, j) << "\"";
                if (i + 1 < m)
                    stream << ", ";
            }
            stream << "]";
            if (j + 1 < n)
                stream << ", ";
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
    template <typename matrix_t>
    std::string visualize_matrix(const matrix_t &A)
    {
        using idx_t = size_type<matrix_t>;
        const idx_t n = std::max(ncols(A), nrows(A));

        if (n > 7)
            return visualize_matrix_text(A);
        else
            return visualize_matrix_table(A);
    }

    //
    // GDB doesn't handle templates well, so we explicitly define some versions of the functions
    // for common template arguments
    //
    std::string visualize_matrix_r(const legacyMatrix<float, Layout::ColMajor> &A)
    {
        return visualize_matrix(A);
    }
    std::string visualize_matrix_d(const legacyMatrix<double, Layout::ColMajor> &A)
    {
        return visualize_matrix(A);
    }
    std::string visualize_matrix_c(const legacyMatrix<std::complex<float>, Layout::ColMajor> &A)
    {
        return visualize_matrix(A);
    }
    std::string visualize_matrix_z(const legacyMatrix<std::complex<double>, Layout::ColMajor> &A)
    {
        return visualize_matrix(A);
    }
    std::string visualize_rowmajormatrix_r(const legacyMatrix<float, Layout::RowMajor> &A)
    {
        return visualize_matrix(A);
    }
    std::string visualize_rowmajormatrix_d(const legacyMatrix<double, Layout::RowMajor> &A)
    {
        return visualize_matrix(A);
    }
    std::string visualize_rowmajormatrix_c(const legacyMatrix<std::complex<float>, Layout::RowMajor> &A)
    {
        return visualize_matrix(A);
    }
    std::string visualize_rowmajormatrix_z(const legacyMatrix<std::complex<double>, Layout::RowMajor> &A)
    {
        return visualize_matrix(A);
    }

}

#endif // TLAPACK_DEBUG_UTILS_HH