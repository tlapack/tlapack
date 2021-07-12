# Example: fortranWrapper_ssymm

In this example, we compute _C - AB_ using matrices A, B and C.

- A is a m-by-m symmetric matrix filled with random numbers.
- B is a m-by-n identity matrix.
- C is a m-by-n matrix. Its _min(m,n)_ first columns are equal to the first columns in matrix 'A'. The remaining entries are zero.

The code uses the routine `ssymm` from [symm.f90](../../src/blas/symm.f90), so that the expected output is _C = 0_. In the final step of the algorithm, we use `snrm2` from [snrm2.f90](../../src/blas/snrm2.f90) to compute the Frobenius norm of C. The norm must be identically null in a successfull run.

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory.

2. Using `make` on the same directory of [example_fortranWrapper_ssymm.f90](example_fortranWrapper_ssymm.f90). In this case, you should edit `make.inc` to set the \<T\>LAPACK include and library directories. After a successful build, the executable will be in the current directory.

## Run

You can run the executable from the command line. The program expects no argument from the input.

---

[Examples](../README.md#fortranWrapper_ssymm)