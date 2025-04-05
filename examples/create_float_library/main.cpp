/// @file create_float_library/main.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <iostream>

#include "tlapack_EigenMatrixXf.hpp"

int main()
{
    Eigen::VectorXf x(3);
    x << 1, 2, 3;

    float sum = tlapack_eigenmatrixxf::asum(x);
    std::cout << "sum is equal to 6? " << (sum == 6 ? "yes" : "no")
              << std::endl;

    return 0;
}