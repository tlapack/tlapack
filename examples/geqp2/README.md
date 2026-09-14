# Example: geqp2

In this example, we compute the QR factorization using column pivoting of a matrix filled with random numbers via Drmac's algorithm.

    A is a m-by-n matrix filled with random numbers.

The code uses the routine tlapack::geqr2 to perform a partial factorization on the left-most k columns, where k is a user input to determine how many columns to not be pivoted, tlapack::unmqr to update the trailing matrix, and tlapack::geqp2 to complete the rank-revealing factorization with column pivoting in place of the trailing matrix, and tlapack::ung2r to generate the m-by-n orthogonal matrix Q. We store R in a n-by-n upper triangular matrix.

To examine the accuracy of the method, we measure ||A - QR||_F / normA​ and, ||I - Q^H Q||_F where F denotes the Frobenius norm, so that the expected output is A-QR = 0. In the final step of the algorithm, we use tlapack::lange to compute the Frobenius norm of the errors. 

To examine the accuracy of the method, we measure
<img src="https://latex.codecogs.com/gif.latex?\|Q^tQ&space;-&space;I\|_F" />
and
<img src="https://latex.codecogs.com/gif.latex?\|QR&space;-&space;A\|_F/\|A\|_F" />,
 In the final step of the algorithm, we use [tlapack::lange] to compute the Frobenius norm of A.

## Build

We provide two options for building this example:

1. Following the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

You will find the executable inside the `build` directory.

2. Using `make` on the same directory of [example_geqp2.cpp](example_geqp2.cpp). In this case, you may need to edit `make.inc` to set the environment variables needed by [Makefile](Makefile). After a successful build, the executable will be in the current directory.

## Run

You can run the executable from the command line. You can pass up to 4 integer arguments to the executable. The first one sets `m` (the number of rows), the second one sets `n` (the number of columns), the third one sets `r` (the rank of the matrix), and the fourth one sets `k` (the number of left-most columns not to be pivoted). Their default values are `m = 7; n = 5; r = 4; k = 2;` for no particular reason.

---

[Examples](../README.md#geqp2)