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

#include "tlapack/base/arrayTraits.hpp"
#include "tlapack/base/workspace.hpp"

namespace starpu {

enum class Operation { Assign, Add, Subtract, Multiply, Divide };

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
}
inline std::ostream& operator<<(std::ostream& out, Operation v)
{
    return out << to_string(v);
}

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

template <class T>
class Matrix {
   public:
    using idx_t = uint32_t;

    struct data {
        starpu_data_handle_t root_handle;
        starpu_data_handle_t handle;
        ~data() { starpu_data_partition_clean(root_handle, 1, &handle); }

        /// Implicit conversion to T
        constexpr operator T() const noexcept
        {
            starpu_data_acquire(handle, STARPU_R);
            const T x = *((T*)starpu_variable_get_local_ptr(handle));
            starpu_data_release(handle);

            return x;
        }

        // Arithmetic operators with assignment

        template <Operation op>
        constexpr data& operate_and_assign(const data& x)
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

    constexpr Matrix(T* ptr, idx_t m, idx_t n, idx_t ld) noexcept
        : is_owner(true)
    {
        starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)ptr,
                                    ld, m, n, sizeof(T));
    }

    constexpr Matrix(T* ptr, idx_t m, idx_t n) noexcept : Matrix(ptr, m, n, m)
    {}

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

    constexpr T operator()(idx_t i, idx_t j) const noexcept
    {
        starpu_data_handle_t root_handle = handle;
        uint32_t pos[2] = {i, j};

        if (starpu_data_get_nb_children(root_handle) > 0) {
            starpu_data_handle_t x0 = starpu_data_get_child(root_handle, 0);
            const idx_t nbx = starpu_matrix_get_nx(x0);
            const idx_t nby =
                starpu_matrix_get_ny(starpu_data_get_child(x0, 0));

            root_handle = starpu_data_get_sub_data(
                root_handle, 2, ix + pos[0] / nbx, iy + pos[1] / nby);

            pos[0] = pos[0] % nbx;
            pos[1] = pos[1] % nby;
        }

        starpu_data_handle_t var_handle[1];

        /* Pick a variable in the matrix */
        struct starpu_data_filter f_var = {
            .filter_func = starpu_matrix_filter_pick_variable,
            .nchildren = 1,
            .get_child_ops = starpu_matrix_filter_pick_variable_child_ops,
            .filter_arg_ptr = (void*)pos};

        starpu_data_partition_plan(root_handle, &f_var, var_handle);
        T value = *((T*)starpu_variable_get_local_ptr(var_handle[0]));
        starpu_data_partition_clean(root_handle, 1, var_handle);

        return value;
    }

    constexpr data operator()(idx_t i, idx_t j) noexcept
    {
        starpu_data_handle_t root_handle = handle;
        uint32_t pos[2] = {i, j};

        if (starpu_data_get_nb_children(root_handle) > 0) {
            starpu_data_handle_t x0 = starpu_data_get_child(root_handle, 0);
            const idx_t nbx = starpu_matrix_get_nx(x0);
            const idx_t nby =
                starpu_matrix_get_ny(starpu_data_get_child(x0, 0));

            root_handle = starpu_data_get_sub_data(
                root_handle, 2, ix + pos[0] / nbx, iy + pos[1] / nby);

            pos[0] = pos[0] % nbx;
            pos[1] = pos[1] % nby;
        }

        starpu_data_handle_t var_handle[1];

        /* Pick a variable in the matrix */
        struct starpu_data_filter f_var = {
            .filter_func = starpu_matrix_filter_pick_variable,
            .nchildren = 1,
            .get_child_ops = starpu_matrix_filter_pick_variable_child_ops,
            .filter_arg_ptr = (void*)pos};

        starpu_data_partition_plan(root_handle, &f_var, var_handle);

        return data(root_handle, var_handle[0]);
    }

    // inline T& operator()(idx_t i, idx_t j) noexcept
    // {
    //     starpu_data_handle_t var_handle[1];

    //     uint32_t pos[2] = {i, j};

    //     /* Pick a variable in the matrix */
    //     struct starpu_data_filter f_var = {
    //         .filter_func = starpu_matrix_filter_pick_variable,
    //         .nchildren = 1,
    //         .get_child_ops = starpu_matrix_filter_pick_variable_child_ops,
    //         .filter_arg_ptr = (void*)pos};

    //     starpu_data_partition_plan(handle, &f_var, var_handle);

    //     return *((T*)starpu_variable_get_local_ptr(var_handle[0]));
    // }

    // inline Matrix get_tile(idx_t i, idx_t j) const noexcept
    // {
    //     return Matrix(starpu_data_get_sub_data(handle, 2, i, j));
    // }

    // inline Matrix block(idx_t i, idx_t j, idx_t m, idx_t n) const noexcept
    // {
    //     Matrix B(handle);

    //     return Matrix(starpu_data_get_sub_data(handle, 2, i, j));
    // }

    // inline Matrix get_tiles(idx_t i, idx_t j, idx_t nx, idx_t ny) const
    // noexcept
    // {
    //     // Matrix tile(data(), nrows() - (nrows() / x_tiles)*nx, ncols() -
    //     (ncols() / y_tiles)*ny, ldim(), nx, ny); _STARPU_CALLOC(tile.handle,
    //     1, sizeof(struct _starpu_data_state)); tile.handle = handle;
    //     (tile.handle)->child_count = 0;

    //     return tile;
    // }

    void create_grid(idx_t nx, idx_t ny) noexcept
    {
        assert(!is_partitioned && "Cannot partition a partitioned matrix");

        /* Split into blocks of complete rows first */
        const struct starpu_data_filter row_split = {
            .filter_func = starpu_matrix_filter_block, .nchildren = nx};

        /* Then split rows into tiles */
        const struct starpu_data_filter col_split = {
            .filter_func = starpu_matrix_filter_vertical_block,
            .nchildren = ny};

        starpu_data_map_filters(handle, 2, &row_split, &col_split);

        is_partitioned = true;
        this->nx = nx;
        this->ny = ny;
    }

    inline constexpr ~Matrix() noexcept
    {
        if (is_partitioned) starpu_data_unpartition(handle, STARPU_MAIN_RAM);
        if (is_owner) starpu_data_unregister(handle);
    }

   private:
    starpu_data_handle_t handle;  ///< Data handle
    const bool is_owner = false;  ///< Whether this object owns the data handle

    bool is_partitioned = false;  ///< Whether the data is partitioned by this
                                  ///< object
    const idx_t ix = 0;  ///< Index of the first tile in the x direction
    const idx_t iy = 0;  ///< Index of the first tile in the y direction
    idx_t nx = 1;        ///< Number of tiles in the x direction
    idx_t ny = 1;        ///< Number of tiles in the y direction

    inline constexpr Matrix(starpu_data_handle_t handle) noexcept
        : handle(handle),
          is_owner(false),
          nx(std::min(1, starpu_data_get_nb_children(handle))),
          ny((starpu_data_get_nb_children(handle) > 0)
                 ? std::min(1,
                            starpu_data_get_nb_children(
                                starpu_data_get_child(handle, 0)))
                 : 1)
    {}

    inline constexpr Matrix(starpu_data_handle_t handle,
                            idx_t ix,
                            idx_t iy,
                            idx_t nx,
                            idx_t ny) noexcept
        : handle(handle), is_owner(false), ix(ix), iy(iy), nx(nx), ny(ny)
    {}
};

}  // namespace starpu

template <class T>
inline std::ostream& operator<<(std::ostream& out, const starpu::Matrix<T>& A)
{
    out << "starpu::Matrix<" << typeid(T).name() << ">( nrows = " << A.nrows()
        << ", ncols = " << A.ncols() << " )";
    if (A.ncols() <= 10) {
        out << "\n";
        for (uint32_t i = 0; i < A.nrows(); ++i) {
            for (uint32_t j = 0; j < A.ncols(); ++j) {
                out << A(i, j) << " ";
            }
            out << "\n";
        }
    }
    return out;
}

namespace tlapack {

template <class T>
struct internal::type_trait<starpu::Matrix<T>, int> {
    using type = T;
};

// -----------------------------------------------------------------------------
// Data descriptors

// Number of rows
template <class T>
inline constexpr auto nrows(const starpu::Matrix<T>& A)
{
    return A.nrows();
}
// Number of columns
template <class T>
inline constexpr auto ncols(const starpu::Matrix<T>& A)
{
    return A.ncols();
}

// Block operations for starpu::Matrix

// #define isSlice(SliceSpec) \
//     std::is_convertible<SliceSpec, std::tuple<uint32_t, uint32_t>>::value

// template <
//     class T,
//     class SliceSpecRow,
//     class SliceSpecCol,
//     typename std::enable_if<isSlice(SliceSpecRow) && isSlice(SliceSpecCol),
//                             int>::type = 0>
// inline constexpr auto slice(const starpu::Matrix<T>& A,
//                             SliceSpecRow&& rows,
//                             SliceSpecCol&& cols) noexcept
// {
//     return starpu::Matrix<T>(
//         A.data() + std::get<0>(rows) + std::get<0>(cols) * A.ldim(),
//         std::get<1>(rows) - std::get<0>(rows),
//         std::get<1>(cols) - std::get<0>(cols), A.ldim());
// }

// #undef isSlice

}  // namespace tlapack

#endif  // TLAPACK_STARPU_HH