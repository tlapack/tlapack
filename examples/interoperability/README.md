# Example: interoperability

In this example, we show how \<T\>LAPACK routines are prepared to handle input matrices and vectors of different classes. This example shows how to use `std::mdspan`, `std::vector`, `Eigen::Matrix` and `tlapack::LegacyVector` classes seamlessly with routines like [tlapack::syr](../../include/blas/syr.hpp), [tlapack::trmm](../../include/blas/trmm.hpp) and [tlapack::getrf](../../include/lapack/potrf.hpp).

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory.

2. Using `make` on the same directory of [example_interoperability.cpp](example_interoperability.cpp). In this case, you may need to edit `make.inc` to set the environment variables needed by [Makefile](Makefile). After a successful build, the executable will be in the current directory.

---

[Examples](../README.md#interoperability)