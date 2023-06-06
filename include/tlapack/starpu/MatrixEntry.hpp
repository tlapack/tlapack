/// @file starpu/MatrixEntry.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_MATRIXENTRY_HH
#define TLAPACK_STARPU_MATRIXENTRY_HH

#include <starpu.h>

#include <tuple>

#include "tlapack/starpu/types.hpp"

namespace tlapack {

namespace internal {
    // for zero types
    template <typename... Types>
    struct real_type_traits;
}  // namespace internal

/// define real_type<> type alias
template <typename... Types>
using real_type = typename internal::real_type_traits<Types..., int>::type;

namespace starpu {

    namespace internal {

        enum class Operation : int {
            Assign = 0,
            Add = 1,
            Subtract = 2,
            Multiply = 3,
            Divide = 4
        };

        /// Return an empty starpu_codelet struct
        constexpr struct starpu_codelet codelet_init() noexcept
        {
            /// TODO: Check that this is correctly initializing values to 0 and
            /// NULL
            return {};
        }

        /**
         * @brief MatrixEntry operation with assignment using two StarPU
         * variable buffers
         *
         * This function is used to perform data operations on MatrixEntry.
         * The interface is suitable for tasks that are submitted to StarPU.
         */
        template <class T, class U, internal::Operation op>
        constexpr void data_op_data(void** buffers, void* args) noexcept
        {
            T* x = (T*)STARPU_VARIABLE_GET_PTR(buffers[0]);
            if constexpr (op == internal::Operation::Assign)
                *x = *((U*)STARPU_VARIABLE_GET_PTR(buffers[1]));
            else if constexpr (op == internal::Operation::Add)
                *x += *((U*)STARPU_VARIABLE_GET_PTR(buffers[1]));
            else if constexpr (op == internal::Operation::Subtract)
                *x -= *((U*)STARPU_VARIABLE_GET_PTR(buffers[1]));
            else if constexpr (op == internal::Operation::Multiply)
                *x *= *((U*)STARPU_VARIABLE_GET_PTR(buffers[1]));
            else if constexpr (op == internal::Operation::Divide)
                *x /= *((U*)STARPU_VARIABLE_GET_PTR(buffers[1]));
        }

        /**
         * @brief MatrixEntry operation with assignment using a StarPU variable
         * buffer and a value
         *
         * This function is used to perform data operations on MatrixEntry.
         * The interface is suitable for tasks that are submitted to StarPU.
         */
        template <class T, class U, internal::Operation op>
        constexpr void data_op_value(void** buffers, void* args) noexcept
        {
            T* x = (T*)STARPU_VARIABLE_GET_PTR(buffers[0]);
            if constexpr (op == internal::Operation::Assign)
                *x = *((U*)args);
            else if constexpr (op == internal::Operation::Add)
                *x += *((U*)args);
            else if constexpr (op == internal::Operation::Subtract)
                *x -= *((U*)args);
            else if constexpr (op == internal::Operation::Multiply)
                *x *= *((U*)args);
            else if constexpr (op == internal::Operation::Divide)
                *x /= *((U*)args);
        }

        /**
         * @brief MatrixEntry operation with assignment using a two matrix
         * entries
         *
         * This function is used to perform data operations on MatrixEntry.
         * The interface is suitable for tasks that are submitted to StarPU.
         */
        template <class T, internal::Operation op>
        void matrixentry_op_matrixentry(void** buffers, void* args) noexcept
        {
            using args_t = std::tuple<idx_t, idx_t, idx_t, idx_t>;

            // get dimensions
            const idx_t& ld = STARPU_MATRIX_GET_LD(buffers[0]);

            // get arguments
            const args_t& cl_args = *(args_t*)args;
            const idx_t& i = std::get<0>(cl_args);
            const idx_t& j = std::get<1>(cl_args);
            const idx_t& k = std::get<2>(cl_args);
            const idx_t& l = std::get<3>(cl_args);

            // get entries
            const uintptr_t& A = STARPU_MATRIX_GET_PTR(buffers[0]);
            T& x = ((T*)A)[i + ld * j];
            const T& y = ((T*)A)[k + ld * l];

            // perform operation
            if constexpr (op == internal::Operation::Assign)
                x = y;
            else if constexpr (op == internal::Operation::Add)
                x += y;
            else if constexpr (op == internal::Operation::Subtract)
                x -= y;
            else if constexpr (op == internal::Operation::Multiply)
                x *= y;
            else if constexpr (op == internal::Operation::Divide)
                x /= y;
        }

    }  // namespace internal

    /**
     * @brief Arithmetic data type used by Matrix
     *
     * This is a wrapper around StarPU variable handles. It is used to
     * perform arithmetic operations on data types stored in StarPU
     * matrices. It uses StarPU tasks to perform the operations.
     *
     * @note Mind that operations between variables may create a large
     * overhead due to the creation of StarPU tasks.
     */
    template <typename T>
    struct MatrixEntry {
        const starpu_data_handle_t root_handle;  ///< Matrix handle
        starpu_data_handle_t handle;             ///< Entry handle
        const idx_t pos[2];  ///< Position of the entry in the matrix

        /// @brief MatrixEntry constructor from a variable handle
        constexpr explicit MatrixEntry(starpu_data_handle_t root_handle,
                                       const idx_t pos[2]) noexcept
            : root_handle(root_handle), pos{pos[0], pos[1]}
        {
            struct starpu_data_filter f_var = {
                .filter_func = starpu_matrix_filter_pick_variable,
                .nchildren = 1,
                .get_child_ops = starpu_matrix_filter_pick_variable_child_ops,
                .filter_arg_ptr = (void*)pos};
            starpu_data_partition_plan(root_handle, &f_var, &handle);
        }

        // Disable copy and move constructors, and copy assignment
        MatrixEntry(MatrixEntry&&) = delete;
        MatrixEntry(const MatrixEntry&) = delete;
        MatrixEntry& operator=(const MatrixEntry&) = delete;

        /// Destructor cleans StarPU partition plan
        ~MatrixEntry() noexcept
        {
            starpu_data_partition_clean(root_handle, 1, &handle);
        }

        /// Implicit conversion to T
        constexpr operator T() const noexcept
        {
            starpu_data_acquire(handle, STARPU_R);
            const T x = *((T*)starpu_variable_get_local_ptr(handle));
            starpu_data_release(handle);

            return x;
        }

        /// Implicit conversion to another object of class MatrixEntry
        template <class U,
                  std::enable_if_t<std::is_same_v<real_type<U>, T>, int> = 0>
        constexpr operator MatrixEntry<U>() const noexcept
        {
            starpu_data_acquire(handle, STARPU_R);
            const T x = *((T*)starpu_variable_get_local_ptr(handle));
            starpu_data_release(handle);

            return x;
        }

        // Arithmetic operators with assignment

        template <class U>
        constexpr MatrixEntry& operator=(MatrixEntry<U>&& x) noexcept
        {
            return operate_and_assign<internal::Operation::Assign, U>(x);
        }
        constexpr MatrixEntry& operator=(const T& x) noexcept
        {
            return operate_and_assign<internal::Operation::Assign, T>(x);
        }

        template <class U>
        constexpr MatrixEntry& operator+=(const MatrixEntry<U>& x) noexcept
        {
            return operate_and_assign<internal::Operation::Add, U>(x);
        }
        constexpr MatrixEntry& operator+=(const T& x) noexcept
        {
            return operate_and_assign<internal::Operation::Add, T>(x);
        }

        template <class U>
        constexpr MatrixEntry& operator-=(const MatrixEntry<U>& x) noexcept
        {
            return operate_and_assign<internal::Operation::Subtract, U>(x);
        }
        constexpr MatrixEntry& operator-=(const T& x) noexcept
        {
            return operate_and_assign<internal::Operation::Subtract, T>(x);
        }

        template <class U>
        constexpr MatrixEntry& operator*=(const MatrixEntry<U>& x) noexcept
        {
            return operate_and_assign<internal::Operation::Multiply, U>(x);
        }
        constexpr MatrixEntry& operator*=(const T& x) noexcept
        {
            return operate_and_assign<internal::Operation::Multiply, T>(x);
        }

        template <class U>
        constexpr MatrixEntry& operator/=(const MatrixEntry<U>& x) noexcept
        {
            return operate_and_assign<internal::Operation::Divide, U>(x);
        }
        constexpr MatrixEntry& operator/=(const T& x) noexcept
        {
            return operate_and_assign<internal::Operation::Divide, T>(x);
        }

        // Other math functions

        constexpr friend real_type<T> abs(const MatrixEntry& x) noexcept
        {
            return abs(T(x));
        }
        constexpr friend T sqrt(const MatrixEntry& x) noexcept
        {
            return sqrt(T(x));
        }

        // Display value in ostream

        constexpr friend std::ostream& operator<<(std::ostream& out,
                                                  const MatrixEntry& x)
        {
            return out << T(x);
        }

       private:
        /// @brief Generates a StarPU codelet for a given operation with a
        /// value
        /// @tparam op Operation to perform
        template <internal::Operation op, class U>
        static constexpr struct starpu_codelet gen_cl_op_value() noexcept
        {
            using internal::Operation;
            struct starpu_codelet cl = internal::codelet_init();

            cl.cpu_funcs[0] = internal::data_op_value<T, U, op>;
            cl.nbuffers = 1;
            cl.modes[0] = (op == Operation::Assign) ? STARPU_W : STARPU_RW;
            cl.name = (op == Operation::Assign)
                          ? "assign_value"
                          : (op == Operation::Add)
                                ? "add_value"
                                : (op == Operation::Subtract)
                                      ? "subtract_value"
                                      : (op == Operation::Multiply)
                                            ? "multiply_value"
                                            : (op == Operation::Divide)
                                                  ? "divide_value"
                                                  : "unknown";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            cl.checked = 1;

            return cl;
        }

        static constexpr const struct starpu_codelet cl_op_value[] = {
            gen_cl_op_value<internal::Operation::Assign, T>(),
            gen_cl_op_value<internal::Operation::Add, T>(),
            gen_cl_op_value<internal::Operation::Subtract, T>(),
            gen_cl_op_value<internal::Operation::Multiply, T>(),
            gen_cl_op_value<internal::Operation::Divide, T>()};
        static constexpr const struct starpu_codelet cl_op_rvalue[] = {
            gen_cl_op_value<internal::Operation::Assign, real_type<T>>(),
            gen_cl_op_value<internal::Operation::Add, real_type<T>>(),
            gen_cl_op_value<internal::Operation::Subtract, real_type<T>>(),
            gen_cl_op_value<internal::Operation::Multiply, real_type<T>>(),
            gen_cl_op_value<internal::Operation::Divide, real_type<T>>()};

        /// @brief Generates a StarPU codelet for a given operation with
        /// another variable
        /// @tparam op Operation to perform
        template <internal::Operation op, class U>
        static constexpr struct starpu_codelet gen_cl_op_data() noexcept
        {
            using internal::Operation;
            struct starpu_codelet cl = internal::codelet_init();

            cl.cpu_funcs[0] = internal::data_op_data<T, U, op>;
            cl.nbuffers = 2;
            cl.modes[0] = (op == Operation::Assign) ? STARPU_W : STARPU_RW;
            cl.modes[1] = STARPU_R;
            cl.name = (op == Operation::Assign)
                          ? "assign_data"
                          : (op == Operation::Add)
                                ? "add_data"
                                : (op == Operation::Subtract)
                                      ? "subtract_data"
                                      : (op == Operation::Multiply)
                                            ? "multiply_data"
                                            : (op == Operation::Divide)
                                                  ? "divide_data"
                                                  : "unknown";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            cl.checked = 1;

            return cl;
        }

        static constexpr const struct starpu_codelet cl_op_data[] = {
            gen_cl_op_data<internal::Operation::Assign, T>(),
            gen_cl_op_data<internal::Operation::Add, T>(),
            gen_cl_op_data<internal::Operation::Subtract, T>(),
            gen_cl_op_data<internal::Operation::Multiply, T>(),
            gen_cl_op_data<internal::Operation::Divide, T>()};
        static constexpr const struct starpu_codelet cl_op_rdata[] = {
            gen_cl_op_data<internal::Operation::Assign, real_type<T>>(),
            gen_cl_op_data<internal::Operation::Add, real_type<T>>(),
            gen_cl_op_data<internal::Operation::Subtract, real_type<T>>(),
            gen_cl_op_data<internal::Operation::Multiply, real_type<T>>(),
            gen_cl_op_data<internal::Operation::Divide, real_type<T>>()};

        /// @brief Generates a StarPU codelet for a given operation  with
        /// entries of a matrix
        /// @tparam op Operation to perform
        template <internal::Operation op>
        static constexpr struct starpu_codelet gen_cl_op_entries() noexcept
        {
            using internal::Operation;
            struct starpu_codelet cl = internal::codelet_init();

            cl.cpu_funcs[0] = internal::matrixentry_op_matrixentry<T, op>;
            cl.nbuffers = 1;
            cl.modes[0] = STARPU_RW;
            cl.name = (op == Operation::Assign)
                          ? "copy_entry"
                          : (op == Operation::Add)
                                ? "add_entry"
                                : (op == Operation::Subtract)
                                      ? "subtract_entry"
                                      : (op == Operation::Multiply)
                                            ? "multiply_entry"
                                            : (op == Operation::Divide)
                                                  ? "divide_entry"
                                                  : "unknown";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            cl.checked = 1;

            return cl;
        }

        static constexpr const struct starpu_codelet cl_op_entries[] = {
            gen_cl_op_entries<internal::Operation::Assign>(),
            gen_cl_op_entries<internal::Operation::Add>(),
            gen_cl_op_entries<internal::Operation::Subtract>(),
            gen_cl_op_entries<internal::Operation::Multiply>(),
            gen_cl_op_entries<internal::Operation::Divide>()};

        /**
         * @brief Applies an operation and assigns
         *
         * Operations: +, -, *, /
         *
         * @tparam op  Operation
         * @param x   Second operand
         * @return MatrixEntry&  Reference to the result
         */
        template <internal::Operation op,
                  class U,
                  std::enable_if_t<(std::is_same_v<U, T> ||
                                    std::is_same_v<U, real_type<T>>),
                                   int> = 0>
        MatrixEntry& operate_and_assign(const MatrixEntry<U>& x)
        {
            // Allocate space for the task
            struct starpu_task* task = starpu_task_create();

            if (this->root_handle == x.root_handle) {
                // If both variables are in the same matrix handle, we must
                // directly perform the operation on the entries of the
                // handle

                using args_t = std::tuple<idx_t, idx_t, idx_t, idx_t>;

                // Allocate space for the arguments
                args_t* args_ptr = new args_t;

                // Initialize arguments
                std::get<0>(*args_ptr) = pos[0];
                std::get<1>(*args_ptr) = pos[1];
                std::get<2>(*args_ptr) = x.pos[0];
                std::get<3>(*args_ptr) = x.pos[1];

                // Initialize task
                task->cl = (struct starpu_codelet*)&(cl_op_entries[int(op)]);
                task->handles[0] = this->root_handle;
                task->cl_arg = (void*)args_ptr;
                task->cl_arg_size = sizeof(args_t);
                task->callback_func = [](void* args) noexcept {
                    delete (args_t*)args;
                };
                task->callback_arg = (void*)args_ptr;
            }
            else {
                // Initialize task
                task->cl = (struct starpu_codelet*)&(
                    std::is_same_v<U, T> ? cl_op_data[int(op)]
                                         : cl_op_rdata[int(op)]);
                task->handles[0] = this->handle;
                task->handles[1] = x.handle;
            }

            // Submit task and check for errors
            const int ret = starpu_task_submit(task);
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
         * @return MatrixEntry&  Reference to the result
         */
        template <internal::Operation op,
                  class U,
                  std::enable_if_t<(std::is_same_v<U, T> ||
                                    std::is_same_v<U, real_type<T>>),
                                   int> = 0>
        MatrixEntry& operate_and_assign(const T& x)
        {
            // Allocate space for the value and initialize it
            T* x_ptr = new T(x);

            // Create and initialize task
            struct starpu_task* task = starpu_task_create();
            task->cl = (struct starpu_codelet*)&(std::is_same_v<U, T>
                                                     ? cl_op_value[int(op)]
                                                     : cl_op_rvalue[int(op)]);
            task->handles[0] = this->handle;
            task->cl_arg = (void*)x_ptr;
            task->cl_arg_size = sizeof(T);
            task->callback_func = [](void* arg) noexcept { delete (T*)arg; };
            task->callback_arg = (void*)x_ptr;

            // Submit task and check for errors
            const int ret = starpu_task_submit(task);
            STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

            return *this;
        }
    };

    // Arithmetic operators with MatrixEntry

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator+(const MatrixEntry<lhs_t>& x,
                             const rhs_t& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) + T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator+(const lhs_t& x,
                             const MatrixEntry<rhs_t>& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) + T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator+(const MatrixEntry<lhs_t>& x,
                             const MatrixEntry<rhs_t>& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) + T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator-(const MatrixEntry<lhs_t>& x,
                             const rhs_t& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) - T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator-(const lhs_t& x,
                             const MatrixEntry<rhs_t>& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) - T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator-(const MatrixEntry<lhs_t>& x,
                             const MatrixEntry<rhs_t>& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) - T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator*(const MatrixEntry<lhs_t>& x,
                             const rhs_t& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) * T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator*(const lhs_t& x,
                             const MatrixEntry<rhs_t>& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) * T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator*(const MatrixEntry<lhs_t>& x,
                             const MatrixEntry<rhs_t>& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) * T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator/(const MatrixEntry<lhs_t>& x,
                             const rhs_t& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) / T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>
    constexpr auto operator/(const lhs_t& x,
                             const MatrixEntry<rhs_t>& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) / T(y);
    }

    template <class lhs_t,
              class rhs_t,
              std::enable_if_t<
                  (std::is_same_v<lhs_t, rhs_t>) ||
                      (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                  int> = 0>

    constexpr auto operator/(const MatrixEntry<lhs_t>& x,
                             const MatrixEntry<rhs_t>& y) noexcept
    {
        using T = scalar_type<lhs_t, rhs_t>;
        return T(x) / T(y);
    }

}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_MATRIXENTRY_HH
