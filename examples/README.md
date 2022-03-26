# \<T\>LAPACK Examples

We provide a few examples to show how to use \<T\>LAPACK. Each example has its own directory with separate build and description. They all assume that \<T\>LAPACK is [installed](../README.md#installation).

This is the list of examples and brief descriptions:

- [geqr2](geqr2/README.md)
  
  Compute the QR factorization of a matrix filled with random numbers.

- [gemm](gemm/README.md)

  Compute _C - AB_ using matrices A, B and C.

- [access_types](access_types/README.md)

  Scale matrices using different access types.

- [cwrapper_gemm](cwrapper_gemm/README.md)

  Compute _C - AB_ using matrices A, B and C with a code written in C.

- [fortranModule_caxpy](fortranModule_caxpy/README.md)

  Compute _c x + y_ using complex matrices x and y and a complex scalar c, with a code written in Fortran90.

- [fortranWrapper_ssymm](fortranWrapper_ssymm/README.md)

  Compute _C - AB_ using matrices A, B and C with a code written in Fortran90.

- [mdspan](mdspan/README.md)

  Combine different matrix layouts from mdspan to compute _C - A B_ using matrices A, B and C.

- [eigen](eigen/README.md)

  Compare the QR factorization from Eigen and \<T\>LAPACK. We use Eigen::Matrix as the data structure.