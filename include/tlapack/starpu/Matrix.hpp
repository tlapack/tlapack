/// @file starpu/Matrix.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_MATRIX_HH
#define TLAPACK_STARPU_MATRIX_HH

#include <starpu.h>

#include <iomanip>
#include <memory>
#include <ostream>
#include <tuple>

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

    using idx_t = uint32_t;

    namespace internal {

        template <class T, bool TisConstType>
        struct EntryAccess;

        enum class Operation : int {
            Assign = 0,
            Add = 1,
            Subtract = 2,
            Multiply = 3,
            Divide = 4
        };

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
                default:
                    return "unknown";
            }
        }

        /// @brief Convert an Operation to an output stream
        inline std::ostream& operator<<(std::ostream& out, Operation v)
        {
            return out << to_string(v);
        }

        /// Return an empty starpu_codelet struct
        constexpr struct starpu_codelet codelet_init() noexcept
        {
            /// TODO: Check that this is correctly initializing values to 0 and
            /// NULL
            return {};
        }

    }  // namespace internal

    namespace cpu {

        /**
         * @brief Data operation with assignment using two StarPU variable
         * buffers
         *
         * This function is used to perform data operations on Matrix<T>::data.
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
         * @brief Data operation with assignment using a StarPU variable buffer
         * and a value
         *
         * This function is used to perform data operations on Matrix<T>::data.
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

        template <class T, internal::Operation op>
        constexpr void matrixentry_op_matrixentry(void** buffers,
                                                  void* args) noexcept
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

    }  // namespace cpu

    namespace internal {

        template <typename T>
        struct data;

        template <class T>
        struct EntryAccess<T, true> {
            // abstract interface
            virtual idx_t nrows() const noexcept = 0;
            virtual idx_t ncols() const noexcept = 0;
            virtual idx_t nblockrows() const noexcept = 0;
            virtual idx_t nblockcols() const noexcept = 0;
            virtual starpu_data_handle_t get_tile_handle(idx_t i,
                                                         idx_t j) noexcept = 0;

            // operator() and operator[]

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

                // A is const, but we need to cast away constness to call
                // get_tile_handle
                auto& A = const_cast<EntryAccess<T, true>&>(*this);

                const starpu_data_handle_t root_handle =
                    A.get_tile_handle(i / mb, j / nb);

                return T(data<T>(root_handle, pos));
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

        template <class T>
        struct EntryAccess<T, false> : public EntryAccess<T, true> {
            using EntryAccess<T, true>::operator();
            using EntryAccess<T, true>::operator[];

            // abstract interface
            virtual idx_t nrows() const noexcept = 0;
            virtual idx_t ncols() const noexcept = 0;
            virtual idx_t nblockrows() const noexcept = 0;
            virtual idx_t nblockcols() const noexcept = 0;
            virtual starpu_data_handle_t get_tile_handle(idx_t i,
                                                         idx_t j) noexcept = 0;

            // operator() and operator[]

            /**
             * @brief Returns a reference to an element of the matrix
             *
             * @param[in] i Row index
             * @param[in] j Column index
             *
             * @return A reference to the element at position (i,j)
             */
            constexpr data<T> operator()(idx_t i, idx_t j) noexcept
            {
                assert((i >= 0 && i < nrows()) && "Row index out of bounds");
                assert((j >= 0 && j < ncols()) && "Column index out of bounds");

                const idx_t mb = nblockrows();
                const idx_t nb = nblockcols();
                const idx_t pos[2] = {i % mb, j % nb};

                const starpu_data_handle_t root_handle =
                    get_tile_handle(i / mb, j / nb);

                return data<T>(root_handle, pos);
            }

            /**
             * @brief Returns a reference to an element of the vector
             *
             * @param[in] i index
             *
             * @return A reference to the element at position i
             */
            constexpr data<T> operator[](idx_t i) noexcept
            {
                assert((nrows() <= 1 || ncols() <= 1) &&
                       "Matrix is not a vector");
                return (nrows() > 1) ? (*this)(i, 0) : (*this)(0, i);
            }
        };

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
        struct data {
            const starpu_data_handle_t root_handle;  ///< Matrix handle
            starpu_data_handle_t handle;             ///< Variable handle
            const idx_t pos[2];

            /// @brief Data constructor from a variable handle
            constexpr explicit data(starpu_data_handle_t root_handle,
                                    const idx_t pos[2]) noexcept
                : root_handle(root_handle), pos{pos[0], pos[1]}
            {
                struct starpu_data_filter f_var = {
                    .filter_func = starpu_matrix_filter_pick_variable,
                    .nchildren = 1,
                    .get_child_ops =
                        starpu_matrix_filter_pick_variable_child_ops,
                    .filter_arg_ptr = (void*)pos};
                starpu_data_partition_plan(root_handle, &f_var, &handle);
            }

            // Disable copy and move constructors, and copy assignment
            data(data&&) = delete;
            data(const data&) = delete;
            data& operator=(const data&) = delete;

            /// Destructor cleans StarPU partition plan
            ~data() noexcept
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

            /// Implicit conversion to T
            template <
                class U,
                std::enable_if_t<std::is_same_v<real_type<U>, T>, int> = 0>
            constexpr operator data<U>() const noexcept
            {
                starpu_data_acquire(handle, STARPU_R);
                const T x = *((T*)starpu_variable_get_local_ptr(handle));
                starpu_data_release(handle);

                return x;
            }

            // Arithmetic operators with assignment

            template <class U>
            constexpr data& operator=(data<U>&& x) noexcept
            {
                return operate_and_assign<Operation::Assign, U>(x);
            }
            constexpr data& operator=(const T& x) noexcept
            {
                return operate_and_assign<Operation::Assign, T>(x);
            }

            template <class U>
            constexpr data& operator+=(const data<U>& x) noexcept
            {
                return operate_and_assign<Operation::Add, U>(x);
            }
            constexpr data& operator+=(const T& x) noexcept
            {
                return operate_and_assign<Operation::Add, T>(x);
            }

            template <class U>
            constexpr data& operator-=(const data<U>& x) noexcept
            {
                return operate_and_assign<Operation::Subtract, U>(x);
            }
            constexpr data& operator-=(const T& x) noexcept
            {
                return operate_and_assign<Operation::Subtract, T>(x);
            }

            template <class U>
            constexpr data& operator*=(const data<U>& x) noexcept
            {
                return operate_and_assign<Operation::Multiply, U>(x);
            }
            constexpr data& operator*=(const T& x) noexcept
            {
                return operate_and_assign<Operation::Multiply, T>(x);
            }

            template <class U>
            constexpr data& operator/=(const data<U>& x) noexcept
            {
                return operate_and_assign<Operation::Divide, U>(x);
            }
            constexpr data& operator/=(const T& x) noexcept
            {
                return operate_and_assign<Operation::Divide, T>(x);
            }

            // Other math functions

            constexpr friend real_type<T> abs(const data& x) noexcept
            {
                return abs(T(x));
            }
            constexpr friend T sqrt(const data& x) noexcept
            {
                return sqrt(T(x));
            }

            // Display value in ostream

            constexpr friend std::ostream& operator<<(std::ostream& out,
                                                      const data& x)
            {
                return out << T(x);
            }

           private:
            /// @brief Generates a StarPU codelet for a given operation with a
            /// value
            /// @tparam op Operation to perform
            template <Operation op, class U>
            static constexpr struct starpu_codelet gen_cl_op_value() noexcept
            {
                struct starpu_codelet cl = codelet_init();

                cl.cpu_funcs[0] = cpu::data_op_value<T, U, op>;
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
                gen_cl_op_value<Operation::Assign, T>(),
                gen_cl_op_value<Operation::Add, T>(),
                gen_cl_op_value<Operation::Subtract, T>(),
                gen_cl_op_value<Operation::Multiply, T>(),
                gen_cl_op_value<Operation::Divide, T>()};
            static constexpr const struct starpu_codelet cl_op_rvalue[] = {
                gen_cl_op_value<Operation::Assign, real_type<T>>(),
                gen_cl_op_value<Operation::Add, real_type<T>>(),
                gen_cl_op_value<Operation::Subtract, real_type<T>>(),
                gen_cl_op_value<Operation::Multiply, real_type<T>>(),
                gen_cl_op_value<Operation::Divide, real_type<T>>()};

            /// @brief Generates a StarPU codelet for a given operation with
            /// another variable
            /// @tparam op Operation to perform
            template <Operation op, class U>
            static constexpr struct starpu_codelet gen_cl_op_data() noexcept
            {
                struct starpu_codelet cl = codelet_init();

                cl.cpu_funcs[0] = cpu::data_op_data<T, U, op>;
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
                gen_cl_op_data<Operation::Assign, T>(),
                gen_cl_op_data<Operation::Add, T>(),
                gen_cl_op_data<Operation::Subtract, T>(),
                gen_cl_op_data<Operation::Multiply, T>(),
                gen_cl_op_data<Operation::Divide, T>()};
            static constexpr const struct starpu_codelet cl_op_rdata[] = {
                gen_cl_op_data<Operation::Assign, real_type<T>>(),
                gen_cl_op_data<Operation::Add, real_type<T>>(),
                gen_cl_op_data<Operation::Subtract, real_type<T>>(),
                gen_cl_op_data<Operation::Multiply, real_type<T>>(),
                gen_cl_op_data<Operation::Divide, real_type<T>>()};

            /// @brief Generates a StarPU codelet for a given operation  with
            /// entries of a matrix
            /// @tparam op Operation to perform
            template <Operation op>
            static constexpr struct starpu_codelet gen_cl_op_entries() noexcept
            {
                struct starpu_codelet cl = codelet_init();

                cl.cpu_funcs[0] = cpu::matrixentry_op_matrixentry<T, op>;
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
                gen_cl_op_entries<Operation::Assign>(),
                gen_cl_op_entries<Operation::Add>(),
                gen_cl_op_entries<Operation::Subtract>(),
                gen_cl_op_entries<Operation::Multiply>(),
                gen_cl_op_entries<Operation::Divide>()};

            /**
             * @brief Applies an operation and assigns
             *
             * Operations: +, -, *, /
             *
             * @tparam op  Operation
             * @param x   Second operand
             * @return constexpr data&  Reference to the result
             */
            template <Operation op,
                      class U,
                      std::enable_if_t<(std::is_same_v<U, T> ||
                                        std::is_same_v<U, real_type<T>>),
                                       int> = 0>
            data& operate_and_assign(const data<U>& x)
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
                    task->cl =
                        (struct starpu_codelet*)&(cl_op_entries[int(op)]);
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
             * @return constexpr data&  Reference to the result
             */
            template <Operation op,
                      class U,
                      std::enable_if_t<(std::is_same_v<U, T> ||
                                        std::is_same_v<U, real_type<T>>),
                                       int> = 0>
            data& operate_and_assign(const T& x)
            {
                // Allocate space for the value and initialize it
                T* x_ptr = new T(x);

                // Create and initialize task
                struct starpu_task* task = starpu_task_create();
                task->cl = (struct starpu_codelet*)&(
                    std::is_same_v<U, T> ? cl_op_value[int(op)]
                                         : cl_op_rvalue[int(op)]);
                task->handles[0] = this->handle;
                task->cl_arg = (void*)x_ptr;
                task->cl_arg_size = sizeof(T);
                task->callback_func = [](void* arg) noexcept {
                    delete (T*)arg;
                };
                task->callback_arg = (void*)x_ptr;

                // Submit task and check for errors
                const int ret = starpu_task_submit(task);
                STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

                return *this;
            }
        };

        // Arithmetic operators with data on the right

        template <class lhs_t,
                  class rhs_t,
                  std::enable_if_t<
                      (std::is_same_v<lhs_t, rhs_t>) ||
                          (std::is_same_v<real_type<lhs_t>, real_type<rhs_t>>),
                      int> = 0>
        constexpr auto operator+(const data<lhs_t>& x, const rhs_t& y) noexcept
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
        constexpr auto operator+(const lhs_t& x, const data<rhs_t>& y) noexcept
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
        constexpr auto operator+(const data<lhs_t>& x,
                                 const data<rhs_t>& y) noexcept
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
        constexpr auto operator-(const data<lhs_t>& x, const rhs_t& y) noexcept
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
        constexpr auto operator-(const lhs_t& x, const data<rhs_t>& y) noexcept
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
        constexpr auto operator-(const data<lhs_t>& x,
                                 const data<rhs_t>& y) noexcept
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
        constexpr auto operator*(const data<lhs_t>& x, const rhs_t& y) noexcept
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
        constexpr auto operator*(const lhs_t& x, const data<rhs_t>& y) noexcept
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
        constexpr auto operator*(const data<lhs_t>& x,
                                 const data<rhs_t>& y) noexcept
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
        constexpr auto operator/(const data<lhs_t>& x, const rhs_t& y) noexcept
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
        constexpr auto operator/(const lhs_t& x, const data<rhs_t>& y) noexcept
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

        constexpr auto operator/(const data<lhs_t>& x,
                                 const data<rhs_t>& y) noexcept
        {
            using T = scalar_type<lhs_t, rhs_t>;
            return T(x) / T(y);
        }

    }  // namespace internal

    template <class T>
    class Matrix : public internal::EntryAccess<T, std::is_const_v<T>> {
       public:
        using internal::EntryAccess<T, std::is_const_v<T>>::operator();
        using internal::EntryAccess<T, std::is_const_v<T>>::operator[];

        // ---------------------------------------------------------------------
        // Constructors and destructor

        /// Create a matrix of size m-by-n from a pointer in main memory
        constexpr Matrix(
            T* ptr, idx_t m, idx_t n, idx_t ld, idx_t nx, idx_t ny) noexcept
            : pHandle(new starpu_data_handle_t(), [](starpu_data_handle_t* h) {
                  starpu_data_unpartition(*h, STARPU_MAIN_RAM);
                  starpu_data_unregister(*h);
                  delete h;
              })
        {
            starpu_matrix_data_register(pHandle.get(), STARPU_MAIN_RAM,
                                        (uintptr_t)ptr, ld, m, n, sizeof(T));
            create_grid(nx, ny);
        }

        /// Create a matrix of size m-by-n from contiguous data in main memory
        constexpr Matrix(T* ptr, idx_t m, idx_t n, idx_t nx, idx_t ny) noexcept
            : Matrix(ptr, m, n, m, nx, ny)
        {}

        /// Create a submatrix from a handle and a grid
        constexpr Matrix(std::shared_ptr<starpu_data_handle_t> pHandle,
                         idx_t ix,
                         idx_t iy,
                         idx_t nx,
                         idx_t ny) noexcept
            : pHandle(pHandle), ix(ix), iy(iy), nx(nx), ny(ny)
        {}

        // Disable copy assignment operator
        Matrix& operator=(const Matrix&) = delete;

        // ---------------------------------------------------------------------
        // Create grid and get tile

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
            const starpu_data_handle_t x0 = starpu_data_get_child(*pHandle, 0);
            return starpu_matrix_get_ny((starpu_data_get_nb_children(x0) > 0)
                                            ? starpu_data_get_child(x0, 0)
                                            : x0);
        }

        /// Get the data handle of a tile in the matrix or the data handle of
        /// the matrix if it is not partitioned
        constexpr starpu_data_handle_t get_tile_handle(idx_t i,
                                                       idx_t j) noexcept
        {
            return starpu_data_get_sub_data(*pHandle, 2, ix + i, iy + j);
        }

        // ---------------------------------------------------------------------
        // Get number of rows and columns

        /**
         * @brief Get the number of rows in the matrix
         *
         * @return Number of rows in the matrix
         */
        idx_t nrows() const noexcept
        {
            const idx_t NX = starpu_data_get_nb_children(*pHandle);
            if (NX <= 1) {
                return starpu_matrix_get_nx(*pHandle);
            }
            else {
                const idx_t nb =
                    starpu_matrix_get_nx(starpu_data_get_child(*pHandle, 0));
                if (ix + nx < NX)
                    return nx * nb;
                else
                    return (nx - 1) * nb +
                           starpu_matrix_get_nx(
                               starpu_data_get_child(*pHandle, NX - 1));
            }
        }

        /**
         * @brief Get the number of columns in the matrix
         *
         * @return Number of columns in the matrix
         */
        idx_t ncols() const noexcept
        {
            const idx_t NX = starpu_data_get_nb_children(*pHandle);
            if (NX <= 1) {
                return starpu_matrix_get_ny(*pHandle);
            }
            else {
                starpu_data_handle_t x0 = starpu_data_get_child(*pHandle, 0);
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

        // ---------------------------------------------------------------------
        // Submatrix creation

        /**
         * @brief Create a submatrix when the matrix is partitioned into tiles
         *
         * @param[in] ix Index of the first tile in x
         * @param[in] iy Index of the first tile in y
         * @param[in] nx Number of tiles in x
         * @param[in] ny Number of tiles in y
         *
         */
        constexpr Matrix<T> get_tiles(idx_t ix,
                                      idx_t iy,
                                      idx_t nx,
                                      idx_t ny) noexcept
        {
            assert(nx >= 0 && ny >= 0 &&
                   "Number of tiles must be positive or 0");
            assert((ix >= 0 && ix + nx <= this->nx) &&
                   "Submatrix out of bounds");
            assert((iy >= 0 && iy + ny <= this->ny) &&
                   "Submatrix out of bounds");

            return Matrix<T>(pHandle, this->ix + ix, this->iy + iy, nx, ny);
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
            assert(nx >= 0 && ny >= 0 &&
                   "Number of tiles must be positive or 0");
            assert((ix >= 0 && ix + nx <= this->nx) &&
                   "Submatrix out of bounds");
            assert((iy >= 0 && iy + ny <= this->ny) &&
                   "Submatrix out of bounds");

            return Matrix<const T>(pHandle, this->ix + ix, this->iy + iy, nx,
                                   ny);
        }

        // ---------------------------------------------------------------------
        // Display matrix in output stream

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

        const idx_t ix = 0;  ///< Index of the first tile in the x direction
        const idx_t iy = 0;  ///< Index of the first tile in the y direction
        idx_t nx = 1;        ///< Number of tiles in the x direction
        idx_t ny = 1;        ///< Number of tiles in the y direction

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
            assert(nx > 0 && ny > 0 && "Number of tiles must be positive");

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
            starpu_data_map_filters(*pHandle, 2, &row_split, &col_split);

            this->nx = nx;
            this->ny = ny;
        }
    };

}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_MATRIX_HH
