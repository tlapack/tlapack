/// @file example_interpretAsMatrix.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// #include <tlapack.hpp>

#include <vector>
#include <iostream>

#include <plugins/tlapack_mdspan.hpp>
#include <tlapack.hpp>

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    using T = float;
    using idx_t = std::size_t;

    using namespace lapack;
    using std::experimental::mdspan;
    using std::experimental::extents;
    
    // constants
    const idx_t n = 10;
    
    // raw data arrays
    T v_[ n ] = {1,2,3,4,5,6,7,8,9,10};
    
    // print array
    for (idx_t i = 0; i < n; i++)
        std::cout << v_[i] << ", ";
    std::cout << std::endl;

    // mdspan arrays
    mdspan< T, extents<10,1> > A{ v_ };
    mdspan< T, extents<1,10> > B{ v_ };
    mdspan< T, extents<10> > v{ v_ };

    std::cout << lange( one_norm, A ) << std::endl;
    std::cout << lange( one_norm, B ) << std::endl;
    std::cout << lange( one_norm, v ) << std::endl;

    return 0;
}
