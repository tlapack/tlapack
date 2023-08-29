# Example: eigen

In this example, we compute the QR factorization of an Eigen::Matrix using \<T\>LAPACK. We then compare with the QR factorization using only Eigen routines.

```cpp
    // Input data
    Eigen::Matrix<float, m, n> A {
        { 1,  2,  3},
        { 4,  5,  6},
        { 7,  8,  9},
        {10, 11, 12},
        {13, 14, 15}
    };
```

The code uses the routine [tlapack::geqr2](../../include/lapack/geqr2.hpp) to perform the complete factorization in place, and [tlapack::ung2r](../../include/lapack/ung2r.hpp) to generate the m-by-n matrix Q. We store R in a n-by-n upper triangular matrix. Also, we use other template BLAS routines.

To examine the accuracy of the method, we measure
<img src="https://latex.codecogs.com/gif.latex?\|Q^tQ&space;-&space;I\|_F" />
and
<img src="https://latex.codecogs.com/gif.latex?\|QR&space;-&space;A\|_F/\|A\|_F" />,
where F denotes the Frobenius norm. For the Frobenius norms we use [tlapack::lange](../../include/lapack/lange.hpp) and [tlapack::lansy](../../include/lapack/lansy.hpp).

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory.

2. Using `make` on the same directory of [example_eigen.cpp](example_eigen.cpp). In this case, you may need to edit `make.inc` to set the environment variables needed by [Makefile](Makefile). After a successful build, the executable will be in the current directory.

## Run

You can run the executable from the command line.

---

[Examples](../README.md#eigen)