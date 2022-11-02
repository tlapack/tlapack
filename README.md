# \<T\>LAPACK
C++ Template Linear Algebra PACKage

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/tlapack/tlapack/blob/master/LICENSE)
[![GitHub Workflow Status](https://github.com/tlapack/tlapack/actions/workflows/cmake.yml/badge.svg)](https://github.com/tlapack/tlapack/actions/workflows/cmake.yml)

## About

First things to know about \<T\>LAPACK:

1. We write &nbsp; \<T\>LAPACK &nbsp; whenever it is possible. This includes all software documentation, discussions, and presentation.
2. We say it &nbsp; T-L-A-PACK &nbsp; .
3. We use &nbsp; `tlapack` &nbsp; for files, folders, and links, to make it more portable and easier to use.

\<T\>LAPACK provides:

+ Precision-neutral function template implementation

*Supported in part by [NSF ACI 2004850](http://www.nsf.gov/awardsearch/showAward?AWD_ID=2004850).*

## Installation

\<T\>LAPACK is built and installed with [CMake](https://cmake.org/).

### Getting CMake

You can either download binaries for the [latest stable](https://cmake.org/download/#latest) or [previous](https://cmake.org/download/#previous) release of CMake,
or build the [current development distribution](https://github.com/Kitware/CMake) from source. CMake is also available in the APT repository on Ubuntu 16.04 or higher.

### Building and Installing \<T\>LAPACK

\<T\>LAPACK can be build following the standard CMake recipe

```sh
mkdir build
cmake -B build -D CMAKE_INSTALL_PREFIX=/path/to/install . # configuration step
cmake --build build                  # build step
cmake --build build --target install # install step
```

### CMake options

Standard environment variables affect CMake.
Some examples are

    CXX                 C++ compiler
    CXXFLAGS            C++ compiler flags
    LDFLAGS             linker flags

The Fortran and C wrappers to \<T\>LAPACK also use, among others,

    CC                  C compiler
    CFLAGS              C compiler flags
    FC                  Fortran compiler
    FFLAGS              Fortran compiler flags

* [This page](https://cmake.org/cmake/help/latest/manual/cmake-env-variables.7.html) lists the environment variables that have special meaning to CMake.

It is also possible to pass variables to CMake during the configuration step using the `-D` flag.
The following example builds \<T\>LAPACK in debug mode inside the directory `build`

```sh
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=Debug .
cmake --build build
```

* [This page](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html) documents variables that are provided by CMake or have meaning to CMake when set by project code.

### \<T\>LAPACK options

Here are the \<T\>LAPACK specific options and their default values

    # Option                            # Default

    BUILD_EXAMPLES                      ON
        
        Build examples
    
    BUILD_TESTING                       ON
    
        Build the testing tree
    
    TLAPACK_BUILD_SINGLE_TESTER         OFF
    
        Build one additional executable that contains all tests
        
    C_WRAPPERS                          OFF
    
        Build and install C wrappers
        
    Fortran_WRAPPERS                    OFF
    
        Build and install Fortran wrappers

    CBLAS_WRAPPERS                      OFF
    
        Build and install CBLAS wrappers to <T>LAPACK
    
    USE_LAPACKPP_WRAPPERS               OFF

        Use LAPACK++ wrappers to link with an optimized LAPACK library.
        Mind that LAPACK++ needs BLAS++.
        Branches compatible with \<T>LAPACK:
            https://bitbucket.org/weslleyspereira/blaspp/branch/tlapack
            https://bitbucket.org/weslleyspereira/lapackpp/branch/tlapack
    
    TLAPACK_INT_T                       int64_t
    
        Type of all non size-related integers in libtlapack_c, libtlapack_cblas, and libtlapack_fortran. It is the type
        used for the array increments, e.g., incx and incy.
        Supported types:
            int, short, long, long long, int8_t, int16_t,
            int32_t, int64_t, int_least8_t, int_least16_t,
            int_least32_t, int_least64_t, int_fast8_t, 
            int_fast16_t, int_fast32_t, int_fast64_t, 
            intmax_t, intptr_t, ptrdiff_t
    
    TLAPACK_SIZE_T                      size_t
    
        Type of all size-relatedintegers in libtlapack_c, libtlapack_cblas, and libtlapack_fortran.
        Supported types:
            int, short, long, long long, int8_t, int16_t,
            int32_t, int64_t, int_least8_t, int_least16_t,
            int_least32_t, int_least64_t, int_fast8_t, 
            int_fast16_t, int_fast32_t, int_fast64_t, 
            intmax_t, intptr_t, ptrdiff_t,
            size_t, uint8_t, uint16_t, uint32_t, uint64_t
    
    BUILD_BLASPP_TESTS                  OFF

        Build BLAS++ tests. Not used if BUILD_TESTING is OFF. If it is ON, you also need to inform blaspp_TEST_DIR,
        which is the path for the test sources of BLAS++.
    
    BUILD_LAPACKPP_TESTS                OFF

        Build LAPACK++ tests. Not used if BUILD_TESTING is OFF. If it is ON, you also need to inform lapackpp_TEST_DIR,
        which is the path for the test sources of LAPACK++.

    BUILD_testBLAS_TESTS                ON

        Build testBLAS tests.

    TLAPACK_NDEBUG                      OFF

        Disable all error checks from <T>LAPACK.

    TLAPACK_CHECK_INPUT                 ON
                                        OFF if TLAPACK_NDEBUG is ON

        <T>LAPACK routines check if the input parameters are illegal.

    TLAPACK_ENABLE_NANCHECK             OFF

        Enable NaN checks for the <T>LAPACK routines.

    TLAPACK_DEFAULT_NANCHECK            ON
                                        OFF if TLAPACK_NDEBUG is ON
                                        OFF if TLAPACK_ENABLE_NANCHECK is OFF

        By default, <T>LAPACK routines check for NaNs as specified in their documentation.

    TLAPACK_ENABLE_INFCHECK             OFF

        Enable Inf checks for the <T>LAPACK routines.

    TLAPACK_DEFAULT_INFCHECK            ON
                                        OFF if TLAPACK_NDEBUG is ON
                                        OFF if TLAPACK_ENABLE_INFCHECK is OFF

        By default, <T>LAPACK routines check for Infs as specified in their documentation.

## Testing

\<T\>LAPACK is currently tested using [testBLAS](https://github.com/tlapack/testBLAS).

## Documentation

+ Run `doxygen docs/Doxyfile` to generate the \<T\>LAPACK documentation via Doxygen.

## License

BSD 3-Clause License

Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.

Copyright (c) 2017-2021, University of Tennessee. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
