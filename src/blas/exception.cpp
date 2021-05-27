// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/exception.hpp"
#include <stdio.h>
#include <stdlib.h>

void blas::error( const char* error_msg, const char* func )
{
#ifdef BLAS_ERROR_ASSERT
    fprintf( stderr, "Error: %s, in function %s\n", error_msg, func );
    abort();
#else
    throw blas::Error( error_msg, func );
#endif
}