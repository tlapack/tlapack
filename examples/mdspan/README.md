# Example: gemm

In this example, we combine different matrix layouts from mdspan to compute _C - A_k B_ using matrices A, A_k, B and C.

- _k <= n/2_
- A is a column-major n-by-n matrix. Its _k_ first are filled with random numbers, and the remaining entries receive `(float) 0xDEADBEEF`. The _k_ last columns are filled with _0_'s or _1_'s. See B above.
- A_k is a column-major n-by-k matrix. It represents the first k columns of A.
- B is a tiled k-by-n identity matrix. It shares data with the last _k_ columns of A.
- C is a m-by-n matrix. Its _k_ first columns are equal to the first columns of A, and the other positions are zero.

The code uses the routine [blas::gemm](../../include/blas/gemm.hpp), so that the expected output is _C = 0_. In the final step of the algorithm, we use [lapack::lange](../../include/lapack/lange.hpp) to compute the Frobenius norm of C. The norm must be identically null.

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory.

2. Using `make` on the same directory of [example_mdspan.cpp](example_mdspan.cpp). In this case, you should edit `make.inc` to set the \<T\>LAPACK include and library directories. After a successful build, the executable will be in the current directory.

## Run

You can run the executable from the command line.

---

[Examples](../README.md#mdspan)