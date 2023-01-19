/// @file cpp_visualizer_example.cpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/debugutils.hpp>
#include <tlapack/lapack/laset.hpp>

#include <memory>
#include <vector>
#include <chrono> // for high_resolution_clock
#include <iostream>

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    typedef float T;

    const int n = 10;
    const T one(1);
    const T zero(0);

    std::unique_ptr<T[]> A_(new T[n * n]);
    tlapack::legacyMatrix<T> A(n, n, &A_[0], n);

    tlapack::laset( tlapack::Uplo::General, zero, one, A );

    std::cout<<"This example is meant to be used with the debugger"<<std::endl<<std::endl;
    std::cout<<"GDB should be able to execute the function `tlapack::print_matrix_r(A)`"<<std::endl;
    std::cout<<"Which will output the matrix A to stdout"<<std::endl<<std::endl;
    std::cout<<"You can also evaluate the expression `tlapack::visualize_matrix_r(A)`"<<std::endl;
    std::cout<<"in the tool vscode-debug-visualizer"<<std::endl<<std::endl;
    std::cout<<"This may require the GDB options `-enable-pretty-printing`, `set print elements 0` and `set print repeats 0`"<<std::endl;

    for( int i = 0; i < n; ++i )
        A(i,i) = i;

    return 0;
}
