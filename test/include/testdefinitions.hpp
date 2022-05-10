/// @file testdefinitions.hpp
/// @brief Definitions for the unit tests
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <legacy_api/legacyArray.hpp>
#include <tlapack.hpp>

namespace tlapack
{

    // 
    // List of matrix types that will be tested
    // 
    using types_to_test = std::tuple<
        legacyMatrix<float, Layout::ColMajor>,
        legacyMatrix<double, Layout::ColMajor>,
        legacyMatrix<std::complex<float>, Layout::ColMajor>,
        legacyMatrix<std::complex<double>, Layout::ColMajor>,
        legacyMatrix<float, Layout::RowMajor>,
        legacyMatrix<double, Layout::RowMajor>,
        legacyMatrix<std::complex<float>, Layout::RowMajor>,
        legacyMatrix<std::complex<double>, Layout::RowMajor>>;
        
    // 
    // The matrix types that will be tested for routines
    // that only accept real matrices
    // 
    using real_types_to_test = std::tuple<
        legacyMatrix<float, Layout::ColMajor>,
        legacyMatrix<double, Layout::ColMajor>,
        legacyMatrix<float, Layout::RowMajor>,
        legacyMatrix<double, Layout::RowMajor>>;

    // 
    // The matrix types that will be tested for routines
    // that only accept complex matrices
    // 
    using complex_types_to_test = std::tuple<
        legacyMatrix<std::complex<float>, Layout::ColMajor>,
        legacyMatrix<std::complex<double>, Layout::ColMajor>,
        legacyMatrix<std::complex<float>, Layout::RowMajor>,
        legacyMatrix<std::complex<double>, Layout::RowMajor>>;

}
