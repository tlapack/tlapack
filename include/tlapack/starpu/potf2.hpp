/// @file starpu/potf2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_POTF2_HH
#define TLAPACK_STARPU_POTF2_HH

#include "tlapack/base/types.hpp"
#include "tlapack/starpu/Matrix.hpp"
#include "tlapack/starpu/tasks.hpp"

namespace tlapack {

template <class uplo_t, class T>
int potf2(uplo_t uplo, starpu::Matrix<T>& A)
{
    using starpu::idx_t;

    // Constants
    const idx_t nx = A.get_nx();
    const idx_t ny = A.get_ny();

    // check arguments
    tlapack_check(nx <= 1 && ny <= 1);

    // Quick return
    if (nx < 1 || ny < 1) return 0;

    // Create info handle
    int info = 0;
    starpu_data_handle_t info_handle;
    starpu_variable_data_register(&info_handle, STARPU_MAIN_RAM,
                                  (uintptr_t)&info, sizeof(int));

    // Insert task to factorize A
    starpu::insert_task_potf2<uplo_t, T>(uplo, A.get_tile_handle(0, 0),
                                         info_handle);

    // Return info
    starpu_data_unregister(info_handle);
    return info;
}

}  // namespace tlapack

#endif  // TLAPACK_POTF2_HH
