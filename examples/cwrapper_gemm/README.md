# Example: cwrapper_gemm

In this example, we compute _C - AB_ using matrices A, B and C.

- A is a m-by-k matrix. Its _min(n,k)_ first columns are filled with random numbers and the remaining entries receive `(float) 0xDEADBEEF`.
- B is a k-by-n identity matrix.
- C is a m-by-n matrix. Its _min(n,k)_ first columns are equal to the first columns in matrix 'A'.

The code uses the routine `dgemm` from [tlapack.h](../../include/tlapack.h), so that the expected output is _C = 0_. In the final step of the algorithm, we use `dnrm2` to compute the Frobenius norm of C. The norm must be identically null in a successfull run.

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory.

2. Using `make` on the same directory of [example_cwrapper_gemm.c](example_cwrapper_gemm.c). In this case, you may need to edit `make.inc` to set the environment variables needed by [Makefile](Makefile). After a successful build, the executable will be in the current directory.

## Run

You can run the executable from the command line. You can pass up to 3 integer arguments to the executable. The first one sets `m`, the second sets `n`, and the third one sets `k`. Their default values are `m = 100; n = 200; k = 50;` for no particular reason.

---

[Examples](../README.md#cwrapper_gemm)