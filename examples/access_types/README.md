# Example: access types

In this example, we scale matrices using [tlapack::lascl](../../include/lapack/lascl.hpp) using different access types.

- A is a 4-by-5 matrix filled with numbers from 1 to 20.

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory.

2. Using `make` on the same directory of [example_accessTypes.cpp](example_accessTypes.cpp). In this case, you may need to edit `make.inc` to set the environment variables needed by [Makefile](Makefile). After a successful build, the executable will be in the current directory.

---

[Examples](../README.md#accessTypes)