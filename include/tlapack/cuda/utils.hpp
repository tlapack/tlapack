/// @file cuda/utils.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_CUDA_UTILS_HH
#define TLAPACK_CUDA_UTILS_HH

#include "tlapack/base/types.hpp"

namespace tlapack {
namespace cuda {

    template <class... Ts>
    struct is_cublas {
        static constexpr bool value = false;
    };

    template <class... Ts>
    constexpr bool is_cublas_v = is_cublas<Ts..., int>::value;

    template <class T>
    struct is_cublas<T,
                     std::enable_if_t<(std::is_same_v<real_type<T>, float> ||
                                       std::is_same_v<real_type<T>, double>),
                                      int>> {
        static constexpr bool value = true;
    };

    template <>
    struct is_cublas<StrongZero, int> {
        static constexpr bool value = true;
    };

    template <class T1, class T2, class... Ts>
    struct is_cublas<T1, T2, Ts...> {
        using T = scalar_type<T1, T2, Ts...>;
        static constexpr bool value = is_cublas<T1, int>::value &&
                                      is_cublas<T2, Ts...>::value &&
                                      std::is_constructible_v<T, T1>;
    };

}  // namespace cuda
}  // namespace tlapack

#endif  // TLAPACK_CUDA_UTILS_HH