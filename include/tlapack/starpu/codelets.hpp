/// @file starpu/codelets.hpp
/// @brief Codelets for StarPU tasks.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_CODELETS_HH
#define TLAPACK_STARPU_CODELETS_HH

#include "tlapack/starpu/functions.hpp"
#include "tlapack/starpu/utils.hpp"

namespace tlapack {
namespace starpu {
    namespace internal {

        // ---------------------------------------------------------------------
        // Functions to generate codelets for BLAS routines

        template <class TA, class TB, class TC, class alpha_t, class beta_t>
        constexpr struct starpu_codelet gen_cl_gemm() noexcept
        {
            struct starpu_codelet cl = codelet_init();
            constexpr bool use_cublas =
                is_cublas_v<TA, TB, TC, alpha_t, beta_t>;

            cl.cpu_funcs[0] = func::gemm<TA, TB, TC, alpha_t, beta_t>;
            if constexpr (use_cublas) {
                cl.cuda_funcs[0] = func::gemm<TA, TB, TC, alpha_t, beta_t, 1>;
                cl.cuda_flags[0] = STARPU_CUDA_ASYNC;
            }
            cl.nbuffers = 3;
            cl.modes[0] = STARPU_R;
            cl.modes[1] = STARPU_R;
            cl.modes[2] = is_same_v<beta_t, StrongZero> ? STARPU_W : STARPU_RW;
            cl.name = "tlapack::starpu::gemm";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            if constexpr (use_cublas) cl.where |= STARPU_CUDA;
            cl.checked = 1;

            return cl;
        }

        template <class TA, class TB, class TC, class alpha_t, class beta_t>
        constexpr struct starpu_codelet gen_cl_symm() noexcept
        {
            struct starpu_codelet cl = codelet_init();

            cl.cpu_funcs[0] = func::symm<TA, TB, TC, alpha_t, beta_t>;
            cl.nbuffers = 3;
            cl.modes[0] = STARPU_R;
            cl.modes[1] = STARPU_R;
            cl.modes[2] = is_same_v<beta_t, StrongZero> ? STARPU_W : STARPU_RW;
            cl.name = "tlapack::starpu::symm";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            cl.checked = 1;

            return cl;
        }

        template <class TA, class TB, class TC, class alpha_t, class beta_t>
        constexpr struct starpu_codelet gen_cl_hemm() noexcept
        {
            struct starpu_codelet cl = codelet_init();

            cl.cpu_funcs[0] = func::hemm<TA, TB, TC, alpha_t, beta_t>;
            cl.nbuffers = 3;
            cl.modes[0] = STARPU_R;
            cl.modes[1] = STARPU_R;
            cl.modes[2] = is_same_v<beta_t, StrongZero> ? STARPU_W : STARPU_RW;
            cl.name = "tlapack::starpu::hemm";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            cl.checked = 1;

            return cl;
        }

        template <class TA, class TC, class alpha_t, class beta_t>
        constexpr struct starpu_codelet gen_cl_syrk() noexcept
        {
            struct starpu_codelet cl = codelet_init();

            cl.cpu_funcs[0] = func::syrk<TA, TC, alpha_t, beta_t>;
            cl.nbuffers = 2;
            cl.modes[0] = STARPU_R;
            cl.modes[1] = is_same_v<beta_t, StrongZero> ? STARPU_W : STARPU_RW;
            cl.name = "tlapack::starpu::syrk";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            cl.checked = 1;

            return cl;
        }

        template <class TA, class TC, class alpha_t, class beta_t>
        constexpr struct starpu_codelet gen_cl_herk() noexcept
        {
            struct starpu_codelet cl = codelet_init();
            constexpr bool use_cublas = is_cublas_v<TA, TC, alpha_t, beta_t>;

            cl.cpu_funcs[0] = func::herk<TA, TC, alpha_t, beta_t>;
            if constexpr (use_cublas) {
                cl.cuda_funcs[0] = func::herk<TA, TC, alpha_t, beta_t, 1>;
                cl.cuda_flags[0] = STARPU_CUDA_ASYNC;
            }
            cl.nbuffers = 2;
            cl.modes[0] = STARPU_R;
            cl.modes[1] = is_same_v<beta_t, StrongZero> ? STARPU_W : STARPU_RW;
            cl.name = "tlapack::starpu::herk";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            if constexpr (use_cublas) cl.where |= STARPU_CUDA;
            cl.checked = 1;

            return cl;
        }

        template <class TA, class TB, class TC, class alpha_t, class beta_t>
        constexpr struct starpu_codelet gen_cl_syr2k() noexcept
        {
            struct starpu_codelet cl = codelet_init();

            cl.cpu_funcs[0] = func::syr2k<TA, TB, TC, alpha_t, beta_t>;
            cl.nbuffers = 3;
            cl.modes[0] = STARPU_R;
            cl.modes[1] = STARPU_R;
            cl.modes[2] = is_same_v<beta_t, StrongZero> ? STARPU_W : STARPU_RW;
            cl.name = "tlapack::starpu::syr2k";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            cl.checked = 1;

            return cl;
        }

        template <class TA, class TB, class TC, class alpha_t, class beta_t>
        constexpr struct starpu_codelet gen_cl_her2k() noexcept
        {
            struct starpu_codelet cl = codelet_init();

            cl.cpu_funcs[0] = func::her2k<TA, TB, TC, alpha_t, beta_t>;
            cl.nbuffers = 3;
            cl.modes[0] = STARPU_R;
            cl.modes[1] = STARPU_R;
            cl.modes[2] = is_same_v<beta_t, StrongZero> ? STARPU_W : STARPU_RW;
            cl.name = "tlapack::starpu::her2k";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            cl.checked = 1;

            return cl;
        }

        template <class TA, class TB, class alpha_t>
        constexpr struct starpu_codelet gen_cl_trmm() noexcept
        {
            struct starpu_codelet cl = codelet_init();

            cl.cpu_funcs[0] = func::trmm<TA, TB, alpha_t>;
            cl.nbuffers = 2;
            cl.modes[0] = STARPU_R;
            cl.modes[1] = STARPU_RW;
            cl.name = "tlapack::starpu::trmm";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            cl.checked = 1;

            return cl;
        }

        template <class TA, class TB, class alpha_t>
        constexpr struct starpu_codelet gen_cl_trsm() noexcept
        {
            struct starpu_codelet cl = codelet_init();
            constexpr bool use_cublas = is_cublas_v<TA, TB, alpha_t>;

            cl.cpu_funcs[0] = func::trsm<TA, TB, alpha_t>;
            if constexpr (use_cublas) {
                cl.cuda_funcs[0] = func::trsm<TA, TB, alpha_t, 1>;
                cl.cuda_flags[0] = STARPU_CUDA_ASYNC;
            }
            cl.nbuffers = 2;
            cl.modes[0] = STARPU_R;
            cl.modes[1] = STARPU_RW;
            cl.name = "tlapack::starpu::trsm";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            if constexpr (use_cublas) cl.where |= STARPU_CUDA;
            cl.checked = 1;

            return cl;
        }

        // ---------------------------------------------------------------------
        // Functions to generate codelets for LAPACK routines

        template <class uplo_t, class T, bool has_info>
        constexpr struct starpu_codelet gen_cl_potrf() noexcept
        {
            struct starpu_codelet cl = codelet_init();
            constexpr bool use_cusolver = is_cusolver_v<T>;

            cl.cpu_funcs[0] = func::potrf<uplo_t, T, has_info>;
            if constexpr (use_cusolver) {
                cl.cuda_funcs[0] = func::potrf<uplo_t, T, has_info, 1>;
                cl.cuda_flags[0] = STARPU_CUDA_ASYNC;
                cl.nbuffers = 2 + (has_info ? 1 : 0);
                cl.modes[1 + (has_info ? 1 : 0)] = starpu_data_access_mode(
                    (int)STARPU_SCRATCH | (int)STARPU_NOFOOTPRINT);
            }
            else {
                cl.nbuffers = 1 + (has_info ? 1 : 0);
            }
            cl.modes[0] = STARPU_RW;
            if constexpr (has_info) cl.modes[1] = STARPU_W;
            cl.name = "tlapack::starpu::potrf";

            // The following lines are needed to make the codelet const
            // See _starpu_codelet_check_deprecated_fields() in StarPU:
            cl.where |= STARPU_CPU;
            if constexpr (use_cusolver) cl.where |= STARPU_CUDA;
            cl.checked = 1;

            return cl;
        }
    }  // namespace internal

    // ---------------------------------------------------------------------
    // Codelets

    /// Codelets for StarPU
    namespace cl {

        template <class TA, class TB, class TC, class alpha_t, class beta_t>
        constexpr const struct starpu_codelet gemm =
            internal::gen_cl_gemm<TA, TB, TC, alpha_t, beta_t>();

        template <class TA, class TB, class TC, class alpha_t, class beta_t>
        constexpr const struct starpu_codelet symm =
            internal::gen_cl_symm<TA, TB, TC, alpha_t, beta_t>();

        template <class TA, class TB, class TC, class alpha_t, class beta_t>
        constexpr const struct starpu_codelet hemm =
            internal::gen_cl_hemm<TA, TB, TC, alpha_t, beta_t>();

        template <class TA, class TC, class alpha_t, class beta_t>
        constexpr const struct starpu_codelet syrk =
            internal::gen_cl_syrk<TA, TC, alpha_t, beta_t>();

        template <class TA, class TC, class alpha_t, class beta_t>
        constexpr const struct starpu_codelet herk =
            internal::gen_cl_herk<TA, TC, alpha_t, beta_t>();

        template <class TA, class TB, class TC, class alpha_t, class beta_t>
        constexpr const struct starpu_codelet syr2k =
            internal::gen_cl_syr2k<TA, TB, TC, alpha_t, beta_t>();

        template <class TA, class TB, class TC, class alpha_t, class beta_t>
        constexpr const struct starpu_codelet her2k =
            internal::gen_cl_her2k<TA, TB, TC, alpha_t, beta_t>();

        template <class TA, class TB, class alpha_t>
        constexpr const struct starpu_codelet trmm =
            internal::gen_cl_trmm<TA, TB, alpha_t>();

        template <class TA, class TB, class alpha_t>
        constexpr const struct starpu_codelet trsm =
            internal::gen_cl_trsm<TA, TB, alpha_t>();

        template <class uplo_t, class T>
        constexpr const struct starpu_codelet potrf =
            internal::gen_cl_potrf<uplo_t, T, true>();

        template <class uplo_t, class T>
        constexpr const struct starpu_codelet potrf_noinfo =
            internal::gen_cl_potrf<uplo_t, T, false>();

    }  // namespace cl

}  // namespace starpu

}  // namespace tlapack

#endif  // TLAPACK_STARPU_CODELETS_HH