# Example: geqr2

In this example, we compute the QR factorization of a matrix filled with random numbers.

- A is a m-by-n matrix filled with random numbers.

The code uses the routine [tlapack::geqr2](../../include/lapack/geqr2.hpp) to perform the complete factorization in place, and [tlapack::ung2r](../../include/lapack/ung2r.hpp) to generate the m-by-n matrix Q. We store R in a n-by-n upper triangular matrix.

To examine the accuracy of the method, we measure
<img src="https://latex.codecogs.com/gif.latex?\|Q^tQ&space;-&space;I\|_F" />
and
<img src="https://latex.codecogs.com/gif.latex?\|QR&space;-&space;A\|_F/\|A\|_F" />,
where F denotes the Frobenius norm.
so that the expected output is C = 0. In the final step of the algorithm, we use [tlapack::nrm2](../../include/blas/nrm2.hpp) to compute the Frobenius norm of C. The norm must be identically null. Set the `verbose` to `true` if you want to see the contents of each matrix in the standard output.

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory.

2. Using `make` on the same directory of [example_geqr2.cpp](example_geqr2.cpp). In this case, you may need to edit `make.inc` to set the environment variables needed by [Makefile](Makefile). After a successful build, the executable will be in the current directory.

## Run

You can run the executable from the command line. You can pass up to 2 integer arguments to the executable. The first one sets `m`, and the second one sets `n`. Their default values are `m = 7; n = 5;` for no particular reason.

---

[Examples](../README.md#geqr2)