# Example: mdspan

In this example, we combine different matrix layouts from mdspan on [tlapack::gemm](../../include/blas/gemm.hpp) and [tlapack::potrf](../../include/lapack/potrf.hpp). 

- Also use the mdspan interfaces of [tlapack::trsm](../../include/blas/trsm.hpp) and [tlapack::lange](../../include/lapack/lange.hpp).
- Use the `submatrix` operation from [base/utils.hpp](../../include/base/utils.hpp)

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
