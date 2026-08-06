/// @file starpu/types.hpp
/// @brief Types for StarPU.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_TYPES_HH
#define TLAPACK_STARPU_TYPES_HH

#include <cstdint>

namespace tlapack {
/** Integration with StarPU
 *
 * Provides a set of tools for integration with the task management library
 * StarPU. See https://starpu.gitlabpages.inria.fr/ for more details about
 * StarPU.
 */
namespace starpu {

    using idx_t = uint32_t;

}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_TYPES_HH