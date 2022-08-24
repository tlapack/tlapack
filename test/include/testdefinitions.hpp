/// @file testdefinitions.hpp
/// @brief Definitions for the unit tests
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TESTDEFINITIONS_HH
#define TLAPACK_TESTDEFINITIONS_HH

#include <tlapack.hpp>

namespace tlapack
{

    // 
    // List of matrix types that will be tested
    // 
    using types_to_test = std::tuple<
        legacyMatrix<float, std::size_t, Layout::ColMajor>,
        legacyMatrix<double, std::size_t, Layout::ColMajor>,
        legacyMatrix<std::complex<float>, std::size_t, Layout::ColMajor>,
        legacyMatrix<std::complex<double>, std::size_t, Layout::ColMajor>,
        legacyMatrix<float, std::size_t, Layout::RowMajor>,
        legacyMatrix<double, std::size_t, Layout::RowMajor>,
        legacyMatrix<std::complex<float>, std::size_t, Layout::RowMajor>,
        legacyMatrix<std::complex<double>, std::size_t, Layout::RowMajor>>;
        
    // 
    // The matrix types that will be tested for routines
    // that only accept real matrices
    // 
    using real_types_to_test = std::tuple<
        legacyMatrix<float, std::size_t, Layout::ColMajor>,
        legacyMatrix<double, std::size_t, Layout::ColMajor>,
        legacyMatrix<float, std::size_t, Layout::RowMajor>,
        legacyMatrix<double, std::size_t, Layout::RowMajor>>;

    // 
    // The matrix types that will be tested for routines
    // that only accept complex matrices
    // 
    using complex_types_to_test = std::tuple<
        legacyMatrix<std::complex<float>, std::size_t, Layout::ColMajor>,
        legacyMatrix<std::complex<double>, std::size_t, Layout::ColMajor>,
        legacyMatrix<std::complex<float>, std::size_t, Layout::RowMajor>,
        legacyMatrix<std::complex<double>, std::size_t, Layout::RowMajor>>;

}

#endif // TLAPACK_TESTDEFINITIONS_HH