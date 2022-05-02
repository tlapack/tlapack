/// @file example_gehd2.cpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tlapack.hpp>
#include <plugins/tlapack_debugutils.hpp>

#include <memory>
#include <vector>
#include <chrono> // for high_resolution_clock
#include <iostream>

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    typedef float T;

    using tlapack::internal::colmajor_matrix;

    const int n = 10;
    const T one(1);
    const T zero(0);

    std::unique_ptr<T[]> _A(new T[n * n]);
    auto A = colmajor_matrix<T>(&_A[0], n, n, n);

    tlapack::laset( tlapack::Uplo::General, zero, one, A );

    std::cout<<"This example is meant to be used with the debugger"<<std::endl<<std::endl;
    std::cout<<"GDB should be able to execute the function `lapack::print_matrix_r(A)`"<<std::endl;
    std::cout<<"Which will output the matrix A to stdout"<<std::endl<<std::endl;
    std::cout<<"You can also evaluate the expression `lapack::visualize_matrix_r(A)`"<<std::endl;
    std::cout<<"in the tool vscode-debug-visualizer"<<std::endl<<std::endl;
    std::cout<<"This may require the GDB options -enable-pretty-printing and set print elements 0"<<std::endl;

    for( int i = 0; i < n; ++i )
        A(i,i) = i;

    return 0;
}
