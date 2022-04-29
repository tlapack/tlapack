# Example: gemm

In this example, we compute _C - AB_ using matrices A, B and C.

- A is a m-by-k matrix. Its _min(n,k)_ first columns are filled with random numbers and the remaining entries receive `(float) 0xDEADBEEF`.
- B is a k-by-n identity matrix.
- C is a m-by-n matrix. Its _min(n,k)_ first columns are equal to the first columns in matrix 'A'.

The code uses the routine [tlapack::gemm](../../include/blas/gemm.hpp), so that the expected output is _C = 0_. In the final step of the algorithm, we use [tlapack::nrm2](../../include/blas/nrm2.hpp) to compute the Frobenius norm of C. The norm must be identically null.

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory. The build system will automatically use [MPFR C++ library](http://www.holoborodko.com/pavel/mpfr/) if CMake finds it installed in your computer.

2. Using `make` on the same directory of [example_gemm.cpp](example_gemm.cpp). In this case, you should edit `make.inc` to set the \<T\>LAPACK include and library directories. After a successful build, the executable will be in the current directory.

## Run

You can run the executable from the command line. You can pass up to 3 integer arguments to the executable. The first one sets `m`, the second sets `n`, and the third one sets `k`. Their default values are `m = 100; n = 200; k = 50;` for no particular reason.

---

[Examples](../README.md#gemm)