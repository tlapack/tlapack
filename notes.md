# Notes

## To discuss

1. Shall we consider sparse data in the library? Where should we consider adding the randomized algorithms? I think another namespace suits well, as done in BLAS++ for the batched algorithms.

2. Intel uses arguments to return the output of `iamax`, `asum`, and others. Should we do that?

3. Intel suggests the use of global variables to turn ON/OFF NAN and INF checks.
   - Discuss about what I did in the `iamax` routine.
   - In my opinion, we can use arguments with default values + global variables to turn on/off the checks.

4. Intel and Mathworks (Bobby) suggest we have GIGO (Garbage In Garbage Out) routines, i.e., that do not check the inputs for invalid values.
   - I think this is a good idea. It is good to have a piece of code that just do the job and, if necessary, couple it with the code that checks.
   - Mathworks (I think) suggests having multiple interfaces, e.g., a GIGO one and a CHECK one. This would avoid the necessity of re-compiling the code.

5. Talk to Mark Gates about:
   - Doxygen configuration in T-LAPACK is based on BLAS++. Is this a problem?
   - T-LAPACK BLAS templates are based on the BLAS++ ones. Is this a problem? Some of the templates were modified to acomodate changes in the T-LAPACK library.
   - T-LAPACK copy some of the utils from BLAS++. Is this OK? Shall we comment about that in place? Shall we add something in the README?
   - T-LAPACK uses BLAS++ and LAPACK++ wrappers by default. Currently, that is the way we link with optimized BLAS and LAPACK libraries.
   - Since oneMKL has a complete C++ interface to BLAS, how does SLATE interfaces oneMKL? Moreover, should T-LAPACK have a direct connection to oneMKL or a connection through SLATE libraries?

6. About the Exception handler:
   - Currently, I am using the same from BLAS++. That is a simple Error class that extends std::exception.
   - The Error class is currently a header `exception.hpp`. My idea is to move the implementation to a source file (This will be the only src of the t-lapack library so far). One motive to do so is that we can enable the user to provide its own implementation of the Error.
   - Should we define a Warning class too? Where should we use warnings instead of Errors?

7. There are 3 different types of using the Fortran 90 interface for T-LAPACK
   1. Fortran program links to the `libctblas.a`, that has the C wrappers, and include the interface `blas.fi`. The program can then call C functions directly using the `iso_c_binding` module.
   2. Fortran program links to the `libctblas.a`, that has the C wrappers, and uses `blas.mod`, that is generated after compiling `blas.f90`. The program calls usual Fortran subroutines using the module `blas.mod`.
   3. Fortran program links to `libftblas.a` and `libctblas.a`, that last one has the Fortran wrappers to the C functions. The program can call the usual Fortran subroutines that will be (possibly) declared using the `external` datatype.

## testBLAS

1. This is supposed to be a library to test C++-based BLAS.

2. We currently have modules with:
   - Tests for corner cases based on the original BLAS papers
   - Tests for INF propagation
   - Tests for NAN propagation

3. It already works with GNU MPFR.

4. All tests for corner cases are generated with a Python script that reads CSV tables with test rules. So, it is easy to change the rule if we want to. We may generated sets with different rules that vigorated in BLAS since its first design.

5. testBLAS is already able to test different BLAS implementations through the link with BLAS++.

## Comparison with BLAS++

Changes in the Exception handling (compatible to the first BLAS papers):
- blas_error_if( n < 0 ) removed from:
asum, axpy, copy, dot, dotu, nrm2, rot, rotm, scal, swap
- "Return immediately if k == 0" was removed from:
gemm, syrk, syr2k, herk, her2k
- "trans == Op::Trans" now provokes error for all data types in her2k and herk
- "trans == Op::Trans" now provokes error for all data types in syr2k and syrk
- rotmg: blas_error_if( *d1 <= 0 );

New things:
- Compatibility with GMP MPFR
- C and Fortran interfaces
- New iamax function inspired in Jim's idea. It never results in NAN or INF for finite input
- blas::size_t and blas::int_t types. The user chooses at compile time. It is compatible with the C and Fortran interfaces

Some of the templates were modified to acomodate changes in the T-LAPACK library

## TO-DO list:

- [x] License: Which licence should we use? Both LAPACK and BLAS++ use the modified BSD license (SPDX-License-Identifier: BSD-3-Clause).
   - Using BSD-3.

- [ ] Change the header of the BLAS templates. It is currently using the information from BLAS++.
   - Let they be like that.

- [ ] Write the remaining Fortran wrappers.
   - There is just a few methods as examples there.

- [ ] Implement the remaining BLAS templates.
   - This includes adding the respective C and Fortran wrappers to the new methods.

- [ ] Choose which LAPACK templates we should implement.

- [ ] Generate a library `testBLAS.a` with the tests. Currently not working like that. Open issue at https://github.com/catchorg/Catch2/issues/2234.

- [ ] Implement the install procedure in CMake

- [ ] Add the error code in the message of Error