# Comparison with BLAS++

Changes in the Exception handling:
- blas_error_if( n < 0 ) removed from:
asum, axpy, copy, dot, dotu, nrm2, rot, rotm, scal, swap
- "Return immediately if k == 0" was removed from:
gemm, syrk, syr2k, herk, her2k
- "trans == Op::Trans" now provokes error for all data types in her2k and herk
- "trans == Op::Trans" now provokes error for all data types in syr2k and syrk
- rotmg: blas_error_if( *d1 <= 0 );

New things:
- Compatibility with GMP MPFR
- C and Fortran wrappers
- New iamax function inspired in Jim's idea. It never results in NAN or INF for finite input
