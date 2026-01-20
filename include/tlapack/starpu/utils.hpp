/// @file starpu/utils.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_CUDA_UTILS_HH
#define TLAPACK_CUDA_UTILS_HH

#include <starpu.h>

#include "tlapack/base/types.hpp"

namespace tlapack {

// C++ standard utils:
using std::enable_if_t;
using std::is_same_v;

namespace starpu {

    /// Type traits for the integration with StarPU
    namespace traits {

        /// Check if a type is supported by cuBLAS
        template <class... Ts>
        struct allow_cublas_trait : std::false_type {};

#ifdef STARPU_USE_CUDA
        template <class T>
        struct allow_cublas_trait<T,
                                  enable_if_t<(is_same_v<real_type<T>, float> ||
                                               is_same_v<real_type<T>, double>),
                                              int>> : std::true_type {};

        template <>
        struct allow_cublas_trait<StrongZero, int> : std::true_type {};

        template <class T1, class T2, class... Ts>
        struct allow_cublas_trait<T1, T2, Ts...> {
            using T = scalar_type<T1, T2, Ts...>;
            static constexpr bool value =
                allow_cublas_trait<T1, int>::value &&
                allow_cublas_trait<T2, Ts...>::value &&
                std::is_constructible_v<T, T1>;
        };
#endif
    }  // namespace traits

    /// Alias for is_cublas<>::value
    template <class... Ts>
    constexpr bool is_cublas_v = traits::allow_cublas_trait<Ts..., int>::value;

    /// True if a type is supported by cuSOLVER
    template <class... Ts>
    constexpr bool is_cusolver_v =
#ifdef STARPU_HAVE_LIBCUSOLVER
        is_cublas_v<Ts...>;
#else
        false;
#endif

#ifdef STARPU_USE_CUDA
    inline cublasOperation_t op2cublas(Op op) noexcept
    {
        switch (op) {
            case Op::NoTrans:
                return CUBLAS_OP_N;
            case Op::Trans:
                return CUBLAS_OP_T;
            case Op::ConjTrans:
                return CUBLAS_OP_C;
            case Op::Conj:
                return CUBLAS_OP_CONJG;
            default:
                return cublasOperation_t(-1);
        }
    }

    inline cublasFillMode_t uplo2cublas(Uplo uplo) noexcept
    {
        switch (uplo) {
            case Uplo::Upper:
                return CUBLAS_FILL_MODE_UPPER;
            case Uplo::Lower:
                return CUBLAS_FILL_MODE_LOWER;
            default:
                return cublasFillMode_t(-1);
        }
    }

    inline cublasDiagType_t diag2cublas(Diag diag) noexcept
    {
        switch (diag) {
            case Diag::NonUnit:
                return CUBLAS_DIAG_NON_UNIT;
            case Diag::Unit:
                return CUBLAS_DIAG_UNIT;
            default:
                return cublasDiagType_t(-1);
        }
    }

    inline cublasSideMode_t side2cublas(Side side) noexcept
    {
        switch (side) {
            case Side::Left:
                return CUBLAS_SIDE_LEFT;
            case Side::Right:
                return CUBLAS_SIDE_RIGHT;
            default:
                return cublasSideMode_t(-1);
        }
    }
#endif
}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_CUDA_UTILS_HH
