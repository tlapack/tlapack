/// @file starpu/Matrix.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_MATRIX_HH
#define TLAPACK_STARPU_MATRIX_HH

#include <starpu.h>

#include <array>
#include <iomanip>
#include <memory>
#include <ostream>

#include "tlapack/starpu/MatrixEntry.hpp"
#include "tlapack/starpu/Tile.hpp"
#include "tlapack/starpu/filters.hpp"

namespace tlapack {
namespace starpu {

    namespace internal {

        /**
         * @brief Class for accessing the elements of a tlapack::starpu::Matrix
         *
         * @details This class is used to implement the operator() and
         * operator[] for the Matrix class. It is not intended to be used
         * directly. Instead, use the Matrix class.
         *
         * @tparam T Type of the elements
         * @tparam TisConstType True if T is const
         */
        template <class T, bool TisConstType>
        struct EntryAccess;

        // EntryAccess for const types
        template <class T>
        struct EntryAccess<T, true> {
            // abstract interface
            virtual idx_t nrows() const noexcept = 0;
            virtual idx_t ncols() const noexcept = 0;
            virtual MatrixEntry<T> map_to_entry(idx_t i, idx_t j) noexcept = 0;

            /**
             * @brief Returns an element of the matrix
             *
             * @param[in] i Row index
             * @param[in] j Column index
             *
             * @return The value of the element at position (i,j)
             */
            constexpr T operator()(idx_t i, idx_t j) const noexcept
            {
                auto& A = const_cast<EntryAccess<T, true>&>(*this);
                return T(A.map_to_entry(i, j));
            }

            /**
             * @brief Returns an element of the vector
             *
             * @param[in] i index
             *
             * @return The value of the element at position i
             */
            constexpr T operator[](idx_t i) const noexcept
            {
                assert((nrows() <= 1 || ncols() <= 1) &&
                       "Matrix is not a vector");
                return (nrows() > 1) ? (*this)(i, 0) : (*this)(0, i);
            }
        };

        // EntryAccess for non-const types
        template <class T>
        struct EntryAccess<T, false> : public EntryAccess<T, true> {
            // abstract interface
            virtual idx_t nrows() const noexcept = 0;
            virtual idx_t ncols() const noexcept = 0;
            virtual MatrixEntry<T> map_to_entry(idx_t i, idx_t j) noexcept = 0;

            /// @copydoc EntryAccess<T, true>::operator()
            using EntryAccess<T, true>::operator();

            /// @copydoc EntryAccess<T, true>::operator[]
            using EntryAccess<T, true>::operator[];

            /**
             * @brief Returns a reference to an element of the matrix
             *
             * @param[in] i Row index
             * @param[in] j Column index
             *
             * @return A reference to the element at position (i,j)
             */
            constexpr MatrixEntry<T> operator()(idx_t i, idx_t j) noexcept
            {
                return map_to_entry(i, j);
            }

            /**
             * @brief Returns a reference to an element of the vector
             *
             * @param[in] i index
             *
             * @return A reference to the element at position i
             */
            constexpr MatrixEntry<T> operator[](idx_t i) noexcept
            {
                assert((nrows() <= 1 || ncols() <= 1) &&
                       "Matrix is not a vector");
                return (nrows() > 1) ? (*this)(i, 0) : (*this)(0, i);
            }
        };

    }  // namespace internal

    /**
     * @brief Class for representing a matrix in StarPU that is split into tiles
     *
     * This class is a wrapper around a StarPU data handle. The grid is created
     * by the StarPU map filters. In order to be able to extract submatrices,
     * this class stores a virtual partition in addition to the StarPU data
     * handle.
     *
     * @tparam T Type of the elements of the matrix
     */
    template <class T>
    class Matrix : public internal::EntryAccess<T, std::is_const_v<T>> {
       public:
        using internal::EntryAccess<T, std::is_const_v<T>>::operator();
        using internal::EntryAccess<T, std::is_const_v<T>>::operator[];

        // ---------------------------------------------------------------------
        // Constructors and destructor

        /** Create a matrix of size m-by-n from a pointer in main memory
         *
         * @param[in] ptr Pointer to the data
         * @param[in] m Number of rows
         * @param[in] n Number of columns
         * @param[in] ld Leading dimension of the matrix
         * @param[in] mt Number of rows in a tile
         * @param[in] nt Number of columns in a tile
         */
        Matrix(T* ptr, idx_t m, idx_t n, idx_t ld, idx_t mt, idx_t nt) noexcept
            : pHandle(new starpu_data_handle_t(), [](starpu_data_handle_t* h) {
                  starpu_data_unpartition(*h, STARPU_MAIN_RAM);
                  starpu_data_unregister(*h);
                  delete h;
              })
        {
            assert(m > 0 && n > 0 && "Invalid matrix size");
            assert(mt <= m && nt <= n && "Invalid tile size");

            nx = (mt == 0) ? 1 : ((m % mt == 0) ? m / mt : m / mt + 1);
            ny = (nt == 0) ? 1 : ((n % nt == 0) ? n / nt : n / nt + 1);

            starpu_matrix_data_register(pHandle.get(), STARPU_MAIN_RAM,
                                        (uintptr_t)ptr, ld, m, n, sizeof(T));
            create_grid(mt, nt);

            starpu_data_handle_t handleN = starpu_data_get_child(
                starpu_data_get_child(*pHandle, nx - 1), ny - 1);
            lastRows = starpu_matrix_get_nx(handleN);
            lastCols = starpu_matrix_get_ny(handleN);
        }

        /** Create a matrix of size m-by-n from contiguous data in main memory
         *
         * @see Matrix(T*, idx_t, idx_t, idx_t, idx_t, idx_t)
         */
        constexpr Matrix(T* ptr, idx_t m, idx_t n, idx_t mt, idx_t nt) noexcept
            : Matrix(ptr, m, n, m, mt, nt)
        {}

        /// Create a submatrix from a handle and a grid
        constexpr Matrix(const std::shared_ptr<starpu_data_handle_t>& pHandle,
                         idx_t ix,
                         idx_t iy,
                         idx_t nx,
                         idx_t ny,
                         idx_t row0,
                         idx_t col0,
                         idx_t lastRows,
                         idx_t lastCols) noexcept
            : pHandle(pHandle),
              ix(ix),
              iy(iy),
              nx(nx),
              ny(ny),
              row0(row0),
              col0(col0),
              lastRows(lastRows),
              lastCols(lastCols)
        {
            assert(ix >= 0 && iy >= 0 && "Invalid tile position");
            assert(nx > 0 && ny > 0 && "Invalid tile size");
            assert(row0 >= 0 && col0 >= 0 && "Invalid tile offset");
            assert(lastRows >= 0 && lastCols >= 0 && "Invalid tile size");
        }

        // Disable copy assignment operator
        Matrix& operator=(const Matrix&) = delete;

        // ---------------------------------------------------------------------
        // Getters

        /// Get number of tiles in x direction
        constexpr idx_t get_nx() const noexcept { return nx; }

        /// Get number of tiles in y direction
        constexpr idx_t get_ny() const noexcept { return ny; }

        /// Get the maximum number of rows of a tile
        constexpr idx_t nblockrows() const noexcept
        {
            return starpu_matrix_get_nx(starpu_data_get_child(*pHandle, 0));
        }

        /// Get the maximum number of columns of a tile
        constexpr idx_t nblockcols() const noexcept
        {
            return starpu_matrix_get_ny(
                starpu_data_get_child(starpu_data_get_child(*pHandle, 0), 0));
        }

        /// Get the data handle of a tile in the matrix or the data handle of
        /// the matrix if it is not partitioned
        Tile tile(idx_t ix, idx_t iy) noexcept
        {
            assert(ix >= 0 && iy >= 0 && ix < nx && iy < ny &&
                   "Invalid tile index");

            starpu_data_handle_t tile_handle = starpu_data_get_sub_data(
                *pHandle, 2, ix + this->ix, iy + this->iy);

            // Collect information about the tile
            idx_t i = 0, j = 0;
            idx_t m = starpu_matrix_get_nx(tile_handle);
            idx_t n = starpu_matrix_get_ny(tile_handle);
            if (ix == 0) {
                i = row0;
                m -= i;
            }
            if (iy == 0) {
                j = col0;
                n -= j;
            }
            if (ix == nx - 1) m = lastRows;
            if (iy == ny - 1) n = lastCols;

            return Tile(tile_handle, i, j, m, n);
        }

        /**
         * @brief Get the number of rows in the matrix
         *
         * @return Number of rows in the matrix
         */
        constexpr idx_t nrows() const noexcept override
        {
            const idx_t mb = nblockrows();
            if (nx == 1) return lastRows;
            if (nx == 2) return (mb - row0) + lastRows;
            return (mb - row0) + (nx - 2) * mb + lastRows;
        }

        /**
         * @brief Get the number of columns in the matrix
         *
         * @return Number of columns in the matrix
         */
        constexpr idx_t ncols() const noexcept override
        {
            const idx_t nb = nblockcols();
            if (ny <= 1) return lastCols;
            if (ny <= 2) return (nb - col0) + lastCols;
            return (nb - col0) + (ny - 2) * nb + lastCols;
        }

        // ---------------------------------------------------------------------
        // Submatrix creation

        /**
         * @brief Create a submatrix from a list of tiles
         *
         * @param[in] ix Index of the first tile in x
         * @param[in] iy Index of the first tile in y
         * @param[in] nx Number of tiles in x
         * @param[in] ny Number of tiles in y
         *
         */
        Matrix<T> get_tiles(idx_t ix, idx_t iy, idx_t nx, idx_t ny) noexcept
        {
            const auto [row0, col0, lastRows, lastCols] =
                _get_tiles_info(ix, iy, nx, ny);

            if (nx == 0) nx = 1;
            if (ny == 0) ny = 1;

            return Matrix<T>(pHandle, this->ix + ix, this->iy + iy, nx, ny,
                             row0, col0, lastRows, lastCols);
        }

        /**
         * @brief Create a submatrix from starting and ending indices
         *
         * @param[in] rowStart Starting row index
         * @param[in] rowEnd Ending row index
         * @param[in] colStart Starting column index
         * @param[in] colEnd Ending column index
         */
        Matrix<T> map_to_tiles(idx_t rowStart,
                               idx_t rowEnd,
                               idx_t colStart,
                               idx_t colEnd) noexcept
        {
            const auto [ix, iy, nx, ny, row0, col0, lastRows, lastCols] =
                _map_to_tiles(rowStart, rowEnd, colStart, colEnd);

            return Matrix<T>(pHandle, this->ix + ix, this->iy + iy, nx, ny,
                             row0, col0, lastRows, lastCols);
        }

        /**
         * @brief Create a const submatrix from a list of tiles
         *
         * @param[in] ix Index of the first tile in x
         * @param[in] iy Index of the first tile in y
         * @param[in] nx Number of tiles in x
         * @param[in] ny Number of tiles in y
         *
         *
         */
        Matrix<const T> get_const_tiles(idx_t ix,
                                        idx_t iy,
                                        idx_t nx,
                                        idx_t ny) const noexcept
        {
            const auto [row0, col0, lastRows, lastCols] =
                _get_tiles_info(ix, iy, nx, ny);

            if (nx == 0) nx = 1;
            if (ny == 0) ny = 1;

            return Matrix<const T>(pHandle, this->ix + ix, this->iy + iy, nx,
                                   ny, row0, col0, lastRows, lastCols);
        }

        /**
         * @brief Create a const submatrix from starting and ending indices
         *
         * @param[in] rowStart Starting row index
         * @param[in] rowEnd Ending row index
         * @param[in] colStart Starting column index
         * @param[in] colEnd Ending column index
         */
        Matrix<const T> map_to_const_tiles(idx_t rowStart,
                                           idx_t rowEnd,
                                           idx_t colStart,
                                           idx_t colEnd) const noexcept
        {
            const auto [ix, iy, nx, ny, row0, col0, lastRows, lastCols] =
                _map_to_tiles(rowStart, rowEnd, colStart, colEnd);

            return Matrix<const T>(pHandle, this->ix + ix, this->iy + iy, nx,
                                   ny, row0, col0, lastRows, lastCols);
        }

        // ---------------------------------------------------------------------
        // Display matrix in output stream

        /**
         * @brief Display matrix in output stream
         *
         * @param[in,out] out Output stream
         * @param[in] A Matrix to display
         */
        friend std::ostream& operator<<(std::ostream& out,
                                        const starpu::Matrix<T>& A)
        {
            out << "starpu::Matrix<" << typeid(T).name()
                << ">( nrows = " << A.nrows() << ", ncols = " << A.ncols()
                << " )";
            if (A.ncols() <= 10) {
                out << std::scientific << std::setprecision(2) << "\n";
                for (idx_t i = 0; i < A.nrows(); ++i) {
                    for (idx_t j = 0; j < A.ncols(); ++j) {
                        const T number = A(i, j);
                        if (abs(number) == -number) out << " ";
                        out << number << " ";
                    }
                    out << "\n";
                }
            }
            return out;
        }

       private:
        std::shared_ptr<starpu_data_handle_t> pHandle;  ///< Data handle

        // Position in the grid
        idx_t ix = 0;  ///< Index of the first tile in the x direction
        idx_t iy = 0;  ///< Index of the first tile in the y direction
        idx_t nx = 1;  ///< Number of tiles in the x direction
        idx_t ny = 1;  ///< Number of tiles in the y direction

        // Position in the first and last tiles of the grid
        idx_t row0 = 0;      ///< Index of the first row in the first tile
        idx_t col0 = 0;      ///< Index of the first column in the first tile
        idx_t lastRows = 0;  ///< Number of rows in the last tile
        idx_t lastCols = 0;  ///< Number of columns in the last tile

        /**
         * @brief Create a grid in the StarPU handle
         *
         * This function creates a grid that partitions the matrix into nx*ny
         * tiles. If the matrix is m-by-n, then every tile (i,j) from 0 <= i <
         * m-1 and 0 <= j < n-1 is a matrix (m/nx)-by-(n/ny). The tiles where i
         * = nx-1 or j = ny-1 are special, as they may have a smaller sizes.
         *
         * @param[in] nx Partitions in x
         * @param[in] ny Partitions in y
         */
        void create_grid(idx_t mt, idx_t nt) noexcept
        {
            /* Split into blocks of complete rows first */
            const struct starpu_data_filter row_split = {
                .filter_func = filter_rows, .nchildren = nx, .filter_arg = mt};

            /* Then split rows into tiles */
            const struct starpu_data_filter col_split = {
                .filter_func = filter_cols, .nchildren = ny, .filter_arg = nt};

            starpu_data_map_filters(*pHandle, 2, &row_split, &col_split);
        }

        /**
         * @brief Gets a reference to the entry (i,j) of the matrix
         *
         * @param[in] i Row index
         * @param[in] j Column index
         *
         * @return MatrixEntry<T> Reference to the entry (i,j)
         */
        MatrixEntry<T> map_to_entry(idx_t i, idx_t j) noexcept override
        {
            const idx_t mb = nblockrows();
            const idx_t nb = nblockcols();

            assert((i >= 0 && i < nrows()) && "Row index out of bounds");
            assert((j >= 0 && j < ncols()) && "Column index out of bounds");

            const idx_t ix = (i + row0) / mb;
            const idx_t iy = (j + col0) / nb;
            const idx_t row = (i + row0) % mb;
            const idx_t col = (j + col0) % nb;

            const idx_t pos[2] = {row, col};

            return MatrixEntry<T>(
                starpu_data_get_sub_data(*pHandle, 2, ix + this->ix,
                                         iy + this->iy),
                pos);
        }

        /**
         * @brief Get the position in the first and last tiles of the grid that
         * should be used in the submatrix.
         *
         * @param[in] ix Index of the first tile in the x direction
         * @param[in] iy Index of the first tile in the y direction
         * @param[in] nx Number of tiles in the x direction
         * @param[in] ny Number of tiles in the y direction
         *
         * @return The array [row0, col0, lastRows, lastCols] to be used in the
         * submatrix.
         */
        std::array<idx_t, 4> _get_tiles_info(idx_t ix,
                                             idx_t iy,
                                             idx_t nx,
                                             idx_t ny) const noexcept
        {
            assert(ix >= 0 && iy >= 0 && ix + nx <= this->nx &&
                   iy + ny <= this->ny && "Invalid tile indices");
            assert(nx >= 0 && ny >= 0 && "Invalid number of tiles");

            const idx_t mb = nblockrows();
            const idx_t nb = nblockcols();

            const idx_t row0 = (ix == 0) ? this->row0 : 0;
            const idx_t col0 = (iy == 0) ? this->col0 : 0;

            idx_t lastRows;
            if (nx == 0)
                lastRows = 0;
            else if (ix + nx == this->nx)
                lastRows = this->lastRows;
            else if (ix + nx == 1)
                lastRows = mb - row0;
            else
                lastRows = mb;

            idx_t lastCols;
            if (ny == 0)
                lastCols = 0;
            else if (iy + ny == this->ny)
                lastCols = this->lastCols;
            else if (iy + ny == 1)
                lastCols = nb - col0;
            else
                lastCols = nb;

            return {row0, col0, lastRows, lastCols};
        }

        /**
         * @brief Get all the virtual grid to be used in the submatrix delimited
         * by [rowStart, rowEnd) x [colStart, colEnd).
         *
         * @param[in] rowStart First row of the submatrix.
         * @param[in] rowEnd Last row of the submatrix.
         * @param[in] colStart First column of the submatrix.
         * @param[in] colEnd Last column of the submatrix.
         *
         * @return The array [ix, iy, nx, ny, row0, col0, lastRows, lastCols]
         */
        std::array<idx_t, 8> _map_to_tiles(idx_t rowStart,
                                           idx_t rowEnd,
                                           idx_t colStart,
                                           idx_t colEnd) const noexcept
        {
            const idx_t mb = this->nblockrows();
            const idx_t nb = this->nblockcols();
            const idx_t nrows = rowEnd - rowStart;
            const idx_t ncols = colEnd - colStart;

            assert(rowStart >= 0 && colStart >= 0 &&
                   "Submatrix starts before the beginning of the matrix");
            assert(rowEnd >= rowStart && colEnd >= colStart &&
                   "Submatrix has negative dimensions");
            assert(rowEnd <= this->nrows() && colEnd <= this->ncols() &&
                   "Submatrix ends after the end of the matrix");

            const idx_t ix = (rowStart + this->row0) / mb;
            const idx_t iy = (colStart + this->col0) / nb;
            const idx_t row0 = (rowStart + this->row0) % mb;
            const idx_t col0 = (colStart + this->col0) % nb;

            const idx_t nx = (nrows == 0) ? 1 : (row0 + nrows - 1) / mb + 1;
            const idx_t ny = (ncols == 0) ? 1 : (col0 + ncols - 1) / nb + 1;

            const idx_t lastRows =
                (nx == 1) ? nrows : (row0 + nrows - 1) % mb + 1;
            const idx_t lastCols =
                (ny == 1) ? ncols : (col0 + ncols - 1) % nb + 1;

            return {ix, iy, nx, ny, row0, col0, lastRows, lastCols};
        }
    };

}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_MATRIX_HH
