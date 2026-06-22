# Example: geqrf_preprocessed

This example follows Golub and Van Loan, Matrix Computations, 4th edition,
section 5.6.1, which discusses consistent systems and transforms that
preserve solvability.  We quote:

If the QR factorization is used to solve Ax=b, then we ordinarily have to
carry out a backsubstitution: Rx = Qᴴb. However, this can be avoided by
"preprocessing" b. Suppose H is a Householder matrix such that H b = β eₙ
where eₙ is the last column of Iₙ. If we compute the QR factorization of
(HA)ᴴ, then A = HᴴRᴴQᴴ and the system transform to Rᴴy = β eₙ where y = Qᴴx.
Since Rᴴ is lower triangular, y = (β/conj(rₙₙ))eₙ and so y = (β/conj(rₙₙ))
Q(:,n).

- A is a m-by-n matrix, with m >= n, filled with random numbers.
- b is a vector of size m filled with random numbers. 

The code uses the routine [tlapack::geqrf](../../include/tlapack/lapack/geqrf.hpp) to perform the complete factorization in place, and [tlapack::larfg](../../include/tlapack/lapack/larfg.hpp) to generate the Householder reflector which is used to preprocess the system.  To apply the Householder reflectors, the code uses [tlapack::larf](../../include/tlapack/lapack/larf.hpp).  

Set the `verbose` to `true` if you want to see the contents of each matrix in the standard output.

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory.

2. Using `make` on the same directory of [example_geqrf_preprocessed.cpp](example_geqrf_preprocessed.cpp). In this case, you may need to edit `make.inc` to set the environment variables needed by [Makefile](Makefile). After a successful build, the executable will be in the current directory.

## Run

You can run the executable from the command line. You can pass up to 2 integer arguments to the executable. The first one sets `m`, and the second one sets `n`. Their default values are `m = 7; n = 5;` for no particular reason.

---

[Examples](../README.md#geqrf_preprocessed)
