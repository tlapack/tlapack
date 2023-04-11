/// @file mdspan.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_HH
#define TLAPACK_STARPU_HH

#include <starpu.h>

#include <ostream>
#include <tuple>

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/workspace.hpp"

namespace tlapack {

namespace starpu {

    enum class Operation { Assign, Add, Subtract, Multiply, Divide };

    /// @brief Convert an Operation to a string
    std::string to_string(Operation v)
    {
        switch (v) {
            case Operation::Assign:
                return "assign";
            case Operation::Add:
                return "add";
            case Operation::Subtract:
                return "subtract";
            case Operation::Multiply:
                return "multiply";
            case Operation::Divide:
                return "divide";
        }
        return "unknown";
    }

    /// @brief Convert an Operation to an output stream
    inline std::ostream& operator<<(std::ostream& out, Operation v)
    {
        return out << to_string(v);
    }

    /**
     * @brief Data operation with assignment using two StarPU variable buffers
     *
     * This function is used to perform data operations on Matrix<T>::data.
     * The interface is suitable for tasks that are submitted to StarPU.
     */
    template <class T, Operation op>
    void data_op_data(void** buffers, void* args)
    {
        T* x = (T*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        if constexpr (op == Operation::Assign)
            *x = *((T*)STARPU_VARIABLE_GET_PTR(buffers[1]));
        else if constexpr (op == Operation::Add)
            *x += *((T*)STARPU_VARIABLE_GET_PTR(buffers[1]));
        else if constexpr (op == Operation::Subtract)
            *x -= *((T*)STARPU_VARIABLE_GET_PTR(buffers[1]));
        else if constexpr (op == Operation::Multiply)
            *x *= *((T*)STARPU_VARIABLE_GET_PTR(buffers[1]));
        else if constexpr (op == Operation::Divide)
            *x /= *((T*)STARPU_VARIABLE_GET_PTR(buffers[1]));
    }

    /**
     * @brief Data operation with assignment using a StarPU variable buffer and
     * a value
     *
     * This function is used to perform data operations on Matrix<T>::data.
     * The interface is suitable for tasks that are submitted to StarPU.
     */
    template <class T, Operation op>
    void data_op_value(void** buffers, void* args)
    {
        T* x = (T*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        if constexpr (op == Operation::Assign)
            *x = *((T*)args);
        else if constexpr (op == Operation::Add)
            *x += *((T*)args);
        else if constexpr (op == Operation::Subtract)
            *x -= *((T*)args);
        else if constexpr (op == Operation::Multiply)
            *x *= *((T*)args);
        else if constexpr (op == Operation::Divide)
            *x /= *((T*)args);
    }

    /**
     * @brief Matrix that abstracts StarPU data handles
     *
     * The Matrix class is a wrapper around StarPU data handles. It provides
     * a simple interface to perform data creation, partitioning, and entry
     * access.
     *
     * @tparam T Data type
     */
    template <class T>
    class Matrix {
       public:
        using idx_t = uint32_t;
        struct data;

        // Constructors

        /// Create a matrix of size m-by-n from a pointer in main memory
        constexpr Matrix(T* ptr, idx_t m, idx_t n, idx_t ld) noexcept
            : is_owner(true)
        {
            starpu_matrix_data_register(&handle, STARPU_MAIN_RAM,
                                        (uintptr_t)ptr, ld, m, n, sizeof(T));
        }

        /// Create a matrix of size m-by-n from contiguous data in main memory
        constexpr Matrix(T* ptr, idx_t m, idx_t n) noexcept
            : Matrix(ptr, m, n, m)
        {}

        // Create grid and get tile

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
        void create_grid(idx_t nx, idx_t ny) noexcept
        {
            assert(!is_partitioned && "Cannot partition a partitioned matrix");
            assert(nx >= 0 && ny >= 0 &&
                   "Number of tiles must be positive or 0");

            /* Split into blocks of complete rows first */
            const struct starpu_data_filter row_split = {
                .filter_func = starpu_matrix_filter_block, .nchildren = nx};

            /* Then split rows into tiles */
            const struct starpu_data_filter col_split = {
                .filter_func = starpu_matrix_filter_vertical_block,
                .nchildren = ny};

            /// TODO: This is not the correct function to use. It only works if
            /// m is a multiple of nx and n is a multiple of ny. We may need to
            /// use another filter.
            starpu_data_map_filters(handle, 2, &row_split, &col_split);

            is_partitioned = true;
            this->nx = nx;
            this->ny = ny;
        }

        /// Get number of tiles in x direction
        constexpr idx_t get_nx() const noexcept { return nx; }

        /// Get number of tiles in y direction
        constexpr idx_t get_ny() const noexcept { return ny; }

        /// Get the maximum number of rows of a tile
        constexpr idx_t nblockrows() const noexcept
        {
            return starpu_matrix_get_nx(
                (starpu_data_get_nb_children(handle) > 0)
                    ? starpu_data_get_child(handle, 0)
                    : handle);
        }

        /// Get the maximum number of columns of a tile
        constexpr idx_t nblockcols() const noexcept
        {
            if (starpu_data_get_nb_children(handle) > 0) {
                const starpu_data_handle_t x0 =
                    starpu_data_get_child(handle, 0);
                return starpu_matrix_get_ny(
                    (starpu_data_get_nb_children(x0) > 0)
                        ? starpu_data_get_child(x0, 0)
                        : x0);
            }
            else
                return starpu_matrix_get_ny(handle);
        }

        /**
         * @brief Create a submatrix when the matrix is partitioned into tiles
         *
         * @param[in] ix Index of the first tile in x
         * @param[in] iy Index of the first tile in y
         * @param[in] nx Number of tiles in x
         * @param[in] ny Number of tiles in y
         *
         */
        constexpr Matrix get_tiles(idx_t ix,
                                   idx_t iy,
                                   idx_t nx,
                                   idx_t ny) noexcept
        {
            assert(is_partitioned && "Matrix is not partitioned");
            assert(nx >= 0 && ny >= 0 &&
                   "Number of tiles must be positive or 0");
            assert((ix >= 0 && ix + nx <= this->nx) &&
                   "Submatrix out of bounds");
            assert((iy >= 0 && iy + ny <= this->ny) &&
                   "Submatrix out of bounds");

            return Matrix(handle, is_partitioned, this->ix + ix, this->iy + iy,
                          nx, ny);
        }

        /**
         * @brief Create a const submatrix when the matrix is partitioned into
         * tiles
         *
         * @param[in] ix Index of the first tile in x
         * @param[in] iy Index of the first tile in y
         * @param[in] nx Number of tiles in x
         * @param[in] ny Number of tiles in y
         *
         */
        constexpr Matrix<const T> get_const_tiles(idx_t ix,
                                                  idx_t iy,
                                                  idx_t nx,
                                                  idx_t ny) const noexcept
        {
            assert(is_partitioned && "Matrix is not partitioned");
            assert(nx >= 0 && ny >= 0 &&
                   "Number of tiles must be positive or 0");
            assert((ix >= 0 && ix + nx <= this->nx) &&
                   "Submatrix out of bounds");
            assert((iy >= 0 && iy + ny <= this->ny) &&
                   "Submatrix out of bounds");

            return Matrix<const T>(handle, is_partitioned, this->ix + ix,
                                   this->iy + iy, nx, ny);
        }

        // Get number of rows and columns

        /**
         * @brief Get the number of rows in the matrix
         *
         * @return Number of rows in the matrix
         */
        idx_t nrows() const noexcept
        {
            const idx_t NX = starpu_data_get_nb_children(handle);
            if (NX <= 1) {
                return starpu_matrix_get_nx(handle);
            }
            else {
                const idx_t nb =
                    starpu_matrix_get_nx(starpu_data_get_child(handle, 0));
                if (ix + nx < NX)
                    return nx * nb;
                else
                    return (nx - 1) * nb +
                           starpu_matrix_get_nx(
                               starpu_data_get_child(handle, NX - 1));
            }
        }

        /**
         * @brief Get the number of columns in the matrix
         *
         * @return Number of columns in the matrix
         */
        idx_t ncols() const noexcept
        {
            const idx_t NX = starpu_data_get_nb_children(handle);
            if (NX <= 1) {
                return starpu_matrix_get_ny(handle);
            }
            else {
                starpu_data_handle_t x0 = starpu_data_get_child(handle, 0);
                const idx_t NY = starpu_data_get_nb_children(x0);
                if (NY <= 1) {
                    return starpu_matrix_get_ny(x0);
                }
                else {
                    const idx_t nb =
                        starpu_matrix_get_ny(starpu_data_get_child(x0, 0));
                    if (iy + ny < NY)
                        return ny * nb;
                    else
                        return (ny - 1) * nb +
                               starpu_matrix_get_ny(
                                   starpu_data_get_child(x0, NY - 1));
                }
            }
        }

        // operator()

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
            assert((i >= 0 && i < nrows()) && "Row index out of bounds");
            assert((j >= 0 && j < ncols()) && "Column index out of bounds");

            const idx_t mb = nblockrows();
            const idx_t nb = nblockcols();
            const idx_t pos[2] = {i % mb, j % nb};

            starpu_data_handle_t root_handle =
                (starpu_data_get_nb_children(handle) > 0)
                    ? starpu_data_get_sub_data(handle, 2, ix + i / mb,
                                               iy + j / nb)
                    : handle;
            starpu_data_handle_t var_handle[1];

            /* Pick a variable in the matrix */
            struct starpu_data_filter f_var = var_filter(pos);
            starpu_data_partition_plan(root_handle, &f_var, var_handle);

            /* Get the value and clean partition plan */
            T value = *((T*)starpu_variable_get_local_ptr(var_handle[0]));
            starpu_data_partition_clean(root_handle, 1, var_handle);

            return value;
        }
        constexpr T operator[](idx_t i) const noexcept
        {
            assert((nrows() <= 1 || ncols() <= 1) && "Matrix is not a vector");

            if (nrows() > 1)
                return (*this)(i, 0);
            else
                return (*this)(0, i);
        }

        /**
         * @brief Returns a reference to an element of the matrix
         *
         * @param[in] i Row index
         * @param[in] j Column index
         *
         * @return A reference to the element at position (i,j)
         */
        template <std::enable_if_t<!std::is_const_v<T>, int> = 0>
        constexpr data operator()(idx_t i, idx_t j) noexcept
        {
            assert((i >= 0 && i < nrows()) && "Row index out of bounds");
            assert((j >= 0 && j < ncols()) && "Column index out of bounds");

            const idx_t mb = nblockrows();
            const idx_t nb = nblockcols();
            const idx_t pos[2] = {i % mb, j % nb};

            starpu_data_handle_t root_handle =
                (starpu_data_get_nb_children(handle) > 0)
                    ? starpu_data_get_sub_data(handle, 2, ix + i / mb,
                                               iy + j / nb)
                    : handle;
            starpu_data_handle_t var_handle[1];

            /* Pick a variable in the matrix */
            struct starpu_data_filter f_var = var_filter(pos);
            starpu_data_partition_plan(root_handle, &f_var, var_handle);

            return data(root_handle, var_handle[0]);
        }
        template <std::enable_if_t<!std::is_const_v<T>, int> = 0>
        constexpr data operator[](idx_t i) noexcept
        {
            assert((nrows() <= 1 || ncols() <= 1) && "Matrix is not a vector");

            if (nrows() > 1)
                return (*this)(i, 0);
            else
                return (*this)(0, i);
        }

        /// @brief Matrix destructor unpartitions and unregisters the data
        /// handle
        constexpr ~Matrix() noexcept
        {
            if (is_owner) {
                if (is_partitioned)
                    starpu_data_unpartition(handle, STARPU_MAIN_RAM);
                starpu_data_unregister(handle);
            }
        }

        // Display matrix in output stream

        friend std::ostream& operator<<(std::ostream& out,
                                        const starpu::Matrix<T>& A)
        {
            using idx_t = typename starpu::Matrix<T>::idx_t;
            out << "starpu::Matrix<" << typeid(T).name()
                << ">( nrows = " << A.nrows() << ", ncols = " << A.ncols()
                << " )";
            if (A.ncols() <= 10) {
                out << "\n";
                for (idx_t i = 0; i < A.nrows(); ++i) {
                    for (idx_t j = 0; j < A.ncols(); ++j) {
                        out << A(i, j) << " ";
                    }
                    out << "\n";
                }
            }
            return out;
        }

       private:
        starpu_data_handle_t handle;  ///< Data handle
        const bool is_owner =
            false;  ///< Whether this object owns the data handle

        bool is_partitioned = false;  ///< Whether the data is partitioned by
                                      ///< this object
        const idx_t ix = 0;  ///< Index of the first tile in the x direction
        const idx_t iy = 0;  ///< Index of the first tile in the y direction
        idx_t nx = 1;        ///< Number of tiles in the x direction
        idx_t ny = 1;        ///< Number of tiles in the y direction

        /// @brief Matrix constructor from a data handle
        constexpr Matrix(starpu_data_handle_t handle) noexcept
            : handle(handle),
              is_owner(false),
              nx(std::min(1, starpu_data_get_nb_children(handle))),
              ny((starpu_data_get_nb_children(handle) > 0)
                     ? std::min(1,
                                starpu_data_get_nb_children(
                                    starpu_data_get_child(handle, 0)))
                     : 1)
        {}

        constexpr Matrix(starpu_data_handle_t handle,
                         bool is_partitioned,
                         idx_t ix,
                         idx_t iy,
                         idx_t nx,
                         idx_t ny) noexcept
            : handle(handle),
              is_owner(false),
              is_partitioned(is_partitioned),
              ix(ix),
              iy(iy),
              nx(nx),
              ny(ny)
        {}

        constexpr struct starpu_data_filter var_filter(
            const idx_t* pos) const noexcept
        {
            return {
                .filter_func = starpu_matrix_filter_pick_variable,
                .nchildren = 1,
                .get_child_ops = starpu_matrix_filter_pick_variable_child_ops,
                .filter_arg_ptr = (void*)pos};
        }
    };

    /**
     * @brief Arithmetic data type used by Matrix
     *
     * This is a wrapper around StarPU variable handles. It is used to perform
     * arithmetic operations on data types stored in StarPU matrices. It uses
     * StarPU tasks to perform the operations.
     *
     * @note Mind that operations between variables may create a large overhead
     * due to the creation of StarPU tasks.
     */
    template <typename T>
    struct Matrix<T>::data {
        starpu_data_handle_t root_handle;  ///< Matrix handle
        starpu_data_handle_t handle;       ///< Variable handle

        /// @brief Data constructor from a variable handle
        constexpr explicit data(starpu_data_handle_t root_handle,
                                starpu_data_handle_t handle) noexcept
            : root_handle(root_handle), handle(handle)
        {}

        // Disable copy and move constructors
        data(data&&) = delete;
        data(const data&) = delete;

        /// Destructor cleans StarPU partition plan
        constexpr ~data() noexcept
        {
            if (handle) starpu_data_partition_clean(root_handle, 1, &handle);
        }

        /// Implicit conversion to T
        constexpr operator T() const noexcept
        {
            starpu_data_acquire(handle, STARPU_R);
            const T x = *((T*)starpu_variable_get_local_ptr(handle));
            starpu_data_release(handle);

            return x;
        }

        // Arithmetic operators with assignment

        /**
         * @brief Applies an operation and assigns
         *
         * Operations: +, -, *, /
         *
         * @tparam op  Operation
         * @param x   Second operand
         * @return constexpr data&  Reference to the result
         */
        template <Operation op>
        constexpr data& operate_and_assign(const data& x) noexcept
        {
            struct starpu_codelet cl = {
                .cpu_funcs = {data_op_data<T, op>},
                .nbuffers = 2,
                .modes = {STARPU_W, STARPU_R},
                .name = (to_string(op) + "_data_" + typeid(T).name()).c_str()};

            const int ret =
                starpu_task_insert(&cl, STARPU_W, this->handle, STARPU_R,
                                   x.handle, STARPU_TASK_SYNCHRONOUS, 1, 0);
            if (ret == -ENODEV) {
                starpu_shutdown();
                exit(77);
            }
            STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

            return *this;
        }

        /**
         * @brief Applies an operation and assigns
         *
         * Operations: +, -, *, /
         *
         * @tparam op  Operation
         * @param x   Second operand
         * @return constexpr data&  Reference to the result
         */
        template <Operation op>
        constexpr data& operate_and_assign(const T& x) noexcept
        {
            struct starpu_codelet cl = {
                .cpu_funcs = {data_op_value<T, op>},
                .nbuffers = 1,
                .modes = {STARPU_W},
                .name = (to_string(op) + "_value_" + typeid(T).name()).c_str()};

            struct starpu_task* task = starpu_task_create();
            task->cl = &cl;
            task->handles[0] = this->handle;
            task->synchronous = 1;
            task->cl_arg = (void*)(&x);
            task->cl_arg_size = sizeof(T);
            const int ret = starpu_task_submit(task);

            // The following code does not work, but it should:
            // int ret = starpu_task_insert(&cl, STARPU_W, this->handle,
            //                              STARPU_VALUE, &x, sizeof(T),
            //                              STARPU_TASK_SYNCHRONOUS, 1, 0);

            if (ret == -ENODEV) {
                starpu_shutdown();
                exit(77);
            }
            STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

            return *this;
        }

        constexpr data& operator=(const data& x) noexcept
        {
            return operate_and_assign<Operation::Assign>(x);
        }
        constexpr data& operator=(const T& x) noexcept
        {
            return operate_and_assign<Operation::Assign>(x);
        }
        constexpr data& operator=(data&& x) noexcept
        {
            return (*this = (const data&)x);
        }

        constexpr data& operator+=(const data& x) noexcept
        {
            return operate_and_assign<Operation::Add>(x);
        }
        constexpr data& operator+=(const T& x) noexcept
        {
            return operate_and_assign<Operation::Add>(x);
        }

        constexpr data& operator-=(const data& x) noexcept
        {
            return operate_and_assign<Operation::Subtract>(x);
        }
        constexpr data& operator-=(const T& x) noexcept
        {
            return operate_and_assign<Operation::Subtract>(x);
        }

        constexpr data& operator*=(const data& x) noexcept
        {
            return operate_and_assign<Operation::Multiply>(x);
        }
        constexpr data& operator*=(const T& x) noexcept
        {
            return operate_and_assign<Operation::Multiply>(x);
        }

        constexpr data& operator/=(const data& x) noexcept
        {
            return operate_and_assign<Operation::Divide>(x);
        }
        constexpr data& operator/=(const T& x) noexcept
        {
            return operate_and_assign<Operation::Divide>(x);
        }

        // Other math functions

        constexpr friend T abs(const data& x) noexcept { return abs(T(x)); }
        constexpr friend T sqrt(const data& x) noexcept { return sqrt(T(x)); }

        // Display value in ostream

        constexpr friend std::ostream& operator<<(std::ostream& out,
                                                  const data& x)
        {
            return out << T(x);
        }
    };

}  // namespace starpu

// -----------------------------------------------------------------------------
// Data descriptors

// Number of rows
template <class T>
constexpr auto nrows(const starpu::Matrix<T>& A)
{
    return A.nrows();
}
// Number of columns
template <class T>
constexpr auto ncols(const starpu::Matrix<T>& A)
{
    return A.ncols();
}
// Size
template <class T>
constexpr auto size(const starpu::Matrix<T>& A)
{
    return A.nrows() * A.ncols();
}

// -----------------------------------------------------------------------------
// Block operations for starpu::Matrix

#define isSlice(SliceSpec) \
    std::is_convertible<SliceSpec, std::tuple<uint32_t, uint32_t>>::value

template <
    class T,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
                            int>::type = 0>
constexpr auto slice(const starpu::Matrix<T>& A,
                     SliceSpecRow&& rows,
                     SliceSpecCol&& cols)
{
    const uint32_t row0 = std::get<0>(rows);
    const uint32_t col0 = std::get<0>(cols);
    const uint32_t row1 = std::get<1>(rows);
    const uint32_t col1 = std::get<1>(cols);
    const uint32_t nrows = row1 - row0;
    const uint32_t ncols = col1 - col0;

    const uint32_t mb = A.nblockrows();
    const uint32_t nb = A.nblockcols();

    tlapack_check(row0 % mb == 0);
    tlapack_check(col0 % nb == 0);
    tlapack_check((nrows % mb == 0) ||
                  (row1 == A.nrows() && ((nrows - 1) % mb == 0)));
    tlapack_check((ncols % nb == 0) ||
                  (col1 == A.ncols() && ((ncols - 1) % nb == 0)));

    return A.get_const_tiles(
        row0 / mb, col0 / nb,
        (nrows == 0) ? 0 : std::max<uint32_t>(nrows / mb, 1),
        (ncols == 0) ? 0 : std::max<uint32_t>(ncols / nb, 1));
}
template <
    class T,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
                            int>::type = 0>
constexpr auto slice(starpu::Matrix<T>& A,
                     SliceSpecRow&& rows,
                     SliceSpecCol&& cols)
{
    const uint32_t row0 = std::get<0>(rows);
    const uint32_t col0 = std::get<0>(cols);
    const uint32_t row1 = std::get<1>(rows);
    const uint32_t col1 = std::get<1>(cols);
    const uint32_t nrows = row1 - row0;
    const uint32_t ncols = col1 - col0;

    const uint32_t mb = A.nblockrows();
    const uint32_t nb = A.nblockcols();

    tlapack_check(row0 % mb == 0);
    tlapack_check(col0 % nb == 0);
    tlapack_check((nrows % mb == 0) ||
                  (row1 == A.nrows() && ((nrows - 1) % mb == 0)));
    tlapack_check((ncols % nb == 0) ||
                  (col1 == A.ncols() && ((ncols - 1) % nb == 0)));

    return A.get_tiles(row0 / mb, col0 / nb,
                       (nrows == 0) ? 0 : std::max<uint32_t>(nrows / mb, 1),
                       (ncols == 0) ? 0 : std::max<uint32_t>(ncols / nb, 1));
}

#undef isSlice

template <class T, class SliceSpec>
constexpr auto slice(const starpu::Matrix<T>& v,
                     SliceSpec&& range,
                     uint32_t colIdx)
{
    return slice(v, std::forward<SliceSpec>(range),
                 std::make_tuple(colIdx, colIdx + 1));
}
template <class T, class SliceSpec>
constexpr auto slice(starpu::Matrix<T>& v, SliceSpec&& range, uint32_t colIdx)
{
    return slice(v, std::forward<SliceSpec>(range),
                 std::make_tuple(colIdx, colIdx + 1));
}

template <class T, class SliceSpec>
constexpr auto slice(const starpu::Matrix<T>& v,
                     uint32_t rowIdx,
                     SliceSpec&& range)
{
    return slice(v, std::make_tuple(rowIdx, rowIdx + 1),
                 std::forward<SliceSpec>(range));
}
template <class T, class SliceSpec>
constexpr auto slice(starpu::Matrix<T>& v, uint32_t rowIdx, SliceSpec&& range)
{
    return slice(v, std::make_tuple(rowIdx, rowIdx + 1),
                 std::forward<SliceSpec>(range));
}

template <class T, class SliceSpec>
constexpr auto slice(const starpu::Matrix<T>& v, SliceSpec&& range)
{
    assert((v.nrows() <= 1 || v.ncols() <= 1) && "Matrix is not a vector");

    if (v.nrows() > 1)
        return slice(v, std::forward<SliceSpec>(range), std::make_tuple(0, 1));
    else
        return slice(v, std::make_tuple(0, 1), std::forward<SliceSpec>(range));
}
template <class T, class SliceSpec>
constexpr auto slice(starpu::Matrix<T>& v, SliceSpec&& range)
{
    assert((v.nrows() <= 1 || v.ncols() <= 1) && "Matrix is not a vector");

    if (v.nrows() > 1)
        return slice(v, std::forward<SliceSpec>(range), std::make_tuple(0, 1));
    else
        return slice(v, std::make_tuple(0, 1), std::forward<SliceSpec>(range));
}

template <class T>
constexpr auto col(const starpu::Matrix<T>& A, uint32_t colIdx)
{
    return slice(A, std::make_tuple(0, A.nrows()),
                 std::make_tuple(colIdx, colIdx + 1));
}
template <class T>
constexpr auto col(starpu::Matrix<T>& A, uint32_t colIdx)
{
    return slice(A, std::make_tuple(0, A.nrows()),
                 std::make_tuple(colIdx, colIdx + 1));
}

template <class T, class SliceSpec>
constexpr auto cols(const starpu::Matrix<T>& A, SliceSpec&& cols)
{
    return slice(A, std::make_tuple(0, A.nrows()),
                 std::forward<SliceSpec>(cols));
}
template <class T, class SliceSpec>
constexpr auto cols(starpu::Matrix<T>& A, SliceSpec&& cols)
{
    return slice(A, std::make_tuple(0, A.nrows()),
                 std::forward<SliceSpec>(cols));
}

template <class T>
constexpr auto row(const starpu::Matrix<T>& A, uint32_t rowIdx)
{
    return slice(A, std::make_tuple(rowIdx, rowIdx + 1),
                 std::make_tuple(0, A.ncols()));
}
template <class T>
constexpr auto row(starpu::Matrix<T>& A, uint32_t rowIdx)
{
    return slice(A, std::make_tuple(rowIdx, rowIdx + 1),
                 std::make_tuple(0, A.ncols()));
}

template <class T, class SliceSpec>
constexpr auto rows(const starpu::Matrix<T>& A, SliceSpec&& rows)
{
    return slice(A, std::forward<SliceSpec>(rows),
                 std::make_tuple(0, A.ncols()));
}
template <class T, class SliceSpec>
constexpr auto rows(starpu::Matrix<T>& A, SliceSpec&& rows)
{
    return slice(A, std::forward<SliceSpec>(rows),
                 std::make_tuple(0, A.ncols()));
}

}  // namespace tlapack

#endif  // TLAPACK_STARPU_HH