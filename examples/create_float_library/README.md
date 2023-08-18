# Example: create_float_library

In this example, we show how to create a static library with single precision from the \<T\>LAPACK library. We use single precision `float` and the data structures from the [Eigen](https://eigen.tuxfamily.org) library, namely, `Eigen::MatrixXf` and `Eigen::VectorXf`.

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the static library and executable inside the `build` directory.

2. Using `make` on the same directory of [tlapack_EigenMatrixXf.hpp](tlapack_EigenMatrixXf.hpp). In this case, you should edit `make.inc` to set the \<T\>LAPACK include and Eigen include directories. After a successful build, the static library and executable will be in the current directory.

## Run

You can run the executable from the command line.

---

[Examples](../README.md#create_float_library)