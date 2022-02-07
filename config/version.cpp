// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <iostream>
#include "version.h"

int main(int argc, char const *argv[])
{
    std::cout << "<T>LAPACK version "
    << TLAPACK_VERSION_MAJOR << "."
    << TLAPACK_VERSION_MINOR << "."
    << TLAPACK_VERSION_PATCH
    << std::endl;
    return 0;
}