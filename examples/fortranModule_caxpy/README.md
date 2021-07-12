# Example: fortranModule_caxpy

In this example, we compute _c x + y_ using complex matrices x and y and a complex scalar c. The scalar is _c = -i_, where i is the square root of -1, and

- x[k] = k * c and
- y[k] = 10*k * c,

for each k from 1 to n. The code uses the routine `caxpy` from [tblas.f90](../../src/tblas.f90). The code prints the error in the 1-norm. One may set the `verbosity` parameter to `.true.` to print all vectors in the default output.

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory.

2. Using `make` on the same directory of [example_fortranModule_caxpy.f90](example_fortranModule_caxpy.f90). In this case, you should edit `make.inc` to set the \<T\>LAPACK include and library directories. After a successful build, the executable will be in the current directory.

** Note that the module file `tblas.mod` is installed in the root of the \<T\>LAPACK include directory. This file is mandatory for both builds. If the module was not correctly installed, one may compile it using the following rules

```Makefile
.o.mod:
    @true
tblas.o: $(tlapack_inc)/tblas.f90 constants.mod
    $(FC) $(FFLAGS) -c -o $@ $<
	rm -f constants.mod constants.o
constants.o: $(tlapack_inc)/blas/constants.f90
    $(FC) $(FFLAGS) -c -o $@ $<
```

in [Makefile](Makefile). In this case, replace `$(tlapack_inc)/tblas.mod` with `tblas.mod` also in [Makefile](Makefile).

## Run

You can run the executable from the command line. The program expects no argument from the input.

---

[Examples](../README.md#fortranModule_caxpy)