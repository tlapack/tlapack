// // Definitions:
// #include "types.hpp"

// namespace blas {

// using Layout = tlapack::Layout;
// using Op = tlapack::Op;
// using Uplo = tlapack::Uplo;
// using Diag = tlapack::Diag;
// using Side = tlapack::Side;

// // =============================================================================
// // Level 1 BLAS

// // -----------------------------------------------------------------------------
// /// @ingroup asum
// float asum(
//     tlapack::size_t n,
//     float const *x, int_t incx );

// /// @ingroup asum
// double asum(
//     tlapack::size_t n,
//     double const *x, int_t incx );

// /// @ingroup asum
// float asum(
//     tlapack::size_t n,
//     std::complex<float> const *x, int_t incx );

// /// @ingroup asum
// double asum(
//     tlapack::size_t n,
//     std::complex<double> const *x, int_t incx );

// // -----------------------------------------------------------------------------
// /// @ingroup axpy
// void axpy(
//     tlapack::size_t n,
//     float alpha,
//     float const *x, int_t incx,
//     float       *y, int_t incy );

// /// @ingroup axpy
// void axpy(
//     tlapack::size_t n,
//     double alpha,
//     double const *x, int_t incx,
//     double       *y, int_t incy );

// /// @ingroup axpy
// void axpy(
//     tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float>       *y, int_t incy );

// /// @ingroup axpy
// void axpy(
//     tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double>       *y, int_t incy );

// // -----------------------------------------------------------------------------
// /// @ingroup copy
// void copy(
//     tlapack::size_t n,
//     float const *x, int_t incx,
//     float       *y, int_t incy );

// /// @ingroup copy
// void copy(
//     tlapack::size_t n,
//     double const *x, int_t incx,
//     double       *y, int_t incy );

// /// @ingroup copy
// void copy(
//     tlapack::size_t n,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float>       *y, int_t incy );

// /// @ingroup copy
// void copy(
//     tlapack::size_t n,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double>       *y, int_t incy );

// // -----------------------------------------------------------------------------
// /// @ingroup dot
// float dot(
//     tlapack::size_t n,
//     float const *x, int_t incx,
//     float const *y, int_t incy );

// /// @ingroup dot
// double dot(
//     tlapack::size_t n,
//     double const *x, int_t incx,
//     double const *y, int_t incy );

// /// @ingroup dot
// std::complex<float> dot(
//     tlapack::size_t n,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float> const *y, int_t incy );

// /// @ingroup dot
// std::complex<double> dot(
//     tlapack::size_t n,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double> const *y, int_t incy );

// // -----------------------------------------------------------------------------
// /// @ingroup dotu
// float dotu(
//     tlapack::size_t n,
//     float const *x, int_t incx,
//     float const *y, int_t incy );

// /// @ingroup dotu
// double dotu(
//     tlapack::size_t n,
//     double const *x, int_t incx,
//     double const *y, int_t incy );

// /// @ingroup dotu
// std::complex<float> dotu(
//     tlapack::size_t n,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float> const *y, int_t incy );

// /// @ingroup dotu
// std::complex<double> dotu(
//     tlapack::size_t n,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double> const *y, int_t incy );

// // -----------------------------------------------------------------------------
// /// @ingroup iamax
// tlapack::size_t iamax(
//     tlapack::size_t n,
//     float const *x, int_t incx );

// /// @ingroup iamax
// tlapack::size_t iamax(
//     tlapack::size_t n,
//     double const *x, int_t incx );

// /// @ingroup iamax
// tlapack::size_t iamax(
//     tlapack::size_t n,
//     std::complex<float> const *x, int_t incx );

// /// @ingroup iamax
// tlapack::size_t iamax(
//     tlapack::size_t n,
//     std::complex<double> const *x, int_t incx );

// // -----------------------------------------------------------------------------
// /// @ingroup nrm2
// float nrm2(
//     tlapack::size_t n,
//     float const *x, int_t incx );

// /// @ingroup nrm2
// double nrm2(
//     tlapack::size_t n,
//     double const *x, int_t incx );

// /// @ingroup nrm2
// float nrm2(
//     tlapack::size_t n,
//     std::complex<float> const *x, int_t incx );

// /// @ingroup nrm2
// double nrm2(
//     tlapack::size_t n,
//     std::complex<double> const *x, int_t incx );

// // -----------------------------------------------------------------------------
// /// @ingroup rot
// void rot(
//     tlapack::size_t n,
//     float *x, int_t incx,
//     float *y, int_t incy,
//     float c,
//     float s );

// /// @ingroup rot
// void rot(
//     tlapack::size_t n,
//     double *x, int_t incx,
//     double *y, int_t incy,
//     double c,
//     double s );

// /// @ingroup rot
// // real cosine, real sine
// void rot(
//     tlapack::size_t n,
//     std::complex<float> *x, int_t incx,
//     std::complex<float> *y, int_t incy,
//     float c,
//     float s );

// /// @ingroup rot
// // real cosine, real sine
// void rot(
//     tlapack::size_t n,
//     std::complex<double> *x, int_t incx,
//     std::complex<double> *y, int_t incy,
//     double c,
//     double s );

// /// @ingroup rot
// // real cosine, complex sine
// void rot(
//     tlapack::size_t n,
//     std::complex<float> *x, int_t incx,
//     std::complex<float> *y, int_t incy,
//     float c,
//     std::complex<float> s );

// /// @ingroup rot
// // real cosine, complex sine
// void rot(
//     tlapack::size_t n,
//     std::complex<double> *x, int_t incx,
//     std::complex<double> *y, int_t incy,
//     double c,
//     std::complex<double> s );

// // -----------------------------------------------------------------------------
// /// @ingroup rotg
// void rotg(
//     float *a,
//     float *b,
//     float *c,
//     float *s );

// /// @ingroup rotg
// void rotg(
//     double *a,
//     double *b,
//     double *c,
//     double *s );

// /// @ingroup rotg
// void rotg(
//     std::complex<float> *a,
//     std::complex<float> *b,  // const in BLAS implementation, oddly
//     float *c,
//     std::complex<float> *s );

// /// @ingroup rotg
// void rotg(
//     std::complex<double> *a,
//     std::complex<double> *b,  // const in BLAS implementation, oddly
//     double *c,
//     std::complex<double> *s );

// // -----------------------------------------------------------------------------
// // only real
// /// @ingroup rotm
// void rotm(
//     tlapack::size_t n,
//     float *x, int_t incx,
//     float *y, int_t incy,
//     float const param[5] );

// /// @ingroup rotm
// void rotm(
//     tlapack::size_t n,
//     double *x, int_t incx,
//     double *y, int_t incy,
//     double const param[5] );

// // -----------------------------------------------------------------------------
// // only real
// /// @ingroup rotmg
// void rotmg(
//     float *d1,
//     float *d2,
//     float *a,
//     float  b,
//     float  param[5] );

// /// @ingroup rotmg
// void rotmg(
//     double *d1,
//     double *d2,
//     double *a,
//     double  b,
//     double  param[5] );

// // -----------------------------------------------------------------------------
// /// @ingroup scal
// void scal(
//     tlapack::size_t n,
//     float alpha,
//     float *x, int_t incx );

// /// @ingroup scal
// void scal(
//     tlapack::size_t n,
//     double alpha,
//     double *x, int_t incx );

// /// @ingroup scal
// void scal(
//     tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> *x, int_t incx );

// /// @ingroup scal
// void scal(
//     tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> *x, int_t incx );

// // -----------------------------------------------------------------------------
// /// @ingroup swap
// void swap(
//     tlapack::size_t n,
//     float *x, int_t incx,
//     float *y, int_t incy );

// /// @ingroup swap
// void swap(
//     tlapack::size_t n,
//     double *x, int_t incx,
//     double *y, int_t incy );

// /// @ingroup swap
// void swap(
//     tlapack::size_t n,
//     std::complex<float> *x, int_t incx,
//     std::complex<float> *y, int_t incy );

// /// @ingroup swap
// void swap(
//     tlapack::size_t n,
//     std::complex<double> *x, int_t incx,
//     std::complex<double> *y, int_t incy );
// // =============================================================================
// // Level 2 BLAS

// // -----------------------------------------------------------------------------
// /// @ingroup gemv
// void gemv(
//     blas::Layout layout,
//     blas::Op trans,
//     tlapack::size_t m, tlapack::size_t n,
//     float alpha,
//     float const *A, int_t lda,
//     float const *x, int_t incx,
//     float beta,
//     float       *y, int_t incy );

// /// @ingroup gemv
// void gemv(
//     blas::Layout layout,
//     blas::Op trans,
//     tlapack::size_t m, tlapack::size_t n,
//     double alpha,
//     double const *A, int_t lda,
//     double const *x, int_t incx,
//     double beta,
//     double       *y, int_t incy );

// /// @ingroup gemv
// void gemv(
//     blas::Layout layout,
//     blas::Op trans,
//     tlapack::size_t m, tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float> beta,
//     std::complex<float>       *y, int_t incy );

// /// @ingroup gemv
// void gemv(
//     blas::Layout layout,
//     blas::Op trans,
//     tlapack::size_t m, tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double> beta,
//     std::complex<double>       *y, int_t incy );

// // -----------------------------------------------------------------------------
// /// @ingroup ger
// void ger(
//     blas::Layout layout,
//     tlapack::size_t m, tlapack::size_t n,
//     float alpha,
//     float const *x, int_t incx,
//     float const *y, int_t incy,
//     float       *A, int_t lda );

// /// @ingroup ger
// void ger(
//     blas::Layout layout,
//     tlapack::size_t m, tlapack::size_t n,
//     double alpha,
//     double const *x, int_t incx,
//     double const *y, int_t incy,
//     double       *A, int_t lda );

// /// @ingroup ger
// void ger(
//     blas::Layout layout,
//     tlapack::size_t m, tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float> const *y, int_t incy,
//     std::complex<float>       *A, int_t lda );

// /// @ingroup ger
// void ger(
//     blas::Layout layout,
//     tlapack::size_t m, tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double> const *y, int_t incy,
//     std::complex<double>       *A, int_t lda );

// // -----------------------------------------------------------------------------
// /// @ingroup geru
// void geru(
//     blas::Layout layout,
//     tlapack::size_t m, tlapack::size_t n,
//     float alpha,
//     float const *x, int_t incx,
//     float const *y, int_t incy,
//     float       *A, int_t lda );

// /// @ingroup geru
// void geru(
//     blas::Layout layout,
//     tlapack::size_t m, tlapack::size_t n,
//     double alpha,
//     double const *x, int_t incx,
//     double const *y, int_t incy,
//     double       *A, int_t lda );

// /// @ingroup geru
// void geru(
//     blas::Layout layout,
//     tlapack::size_t m, tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float> const *y, int_t incy,
//     std::complex<float>       *A, int_t lda );

// /// @ingroup geru
// void geru(
//     blas::Layout layout,
//     tlapack::size_t m, tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double> const *y, int_t incy,
//     std::complex<double>       *A, int_t lda );

// // -----------------------------------------------------------------------------
// /// @ingroup hemv
// void hemv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     float alpha,
//     float const *A, int_t lda,
//     float const *x, int_t incx,
//     float beta,
//     float       *y, int_t incy );

// /// @ingroup hemv
// void hemv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     double alpha,
//     double const *A, int_t lda,
//     double const *x, int_t incx,
//     double beta,
//     double       *y, int_t incy );

// /// @ingroup hemv
// void hemv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float> beta,
//     std::complex<float>       *y, int_t incy );

// /// @ingroup hemv
// void hemv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double> beta,
//     std::complex<double>       *y, int_t incy );

// // -----------------------------------------------------------------------------
// /// @ingroup her
// void her(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     float alpha,
//     float const *x, int_t incx,
//     float       *A, int_t lda );

// /// @ingroup her
// void her(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     double alpha,
//     double const *x, int_t incx,
//     double       *A, int_t lda );

// /// @ingroup her
// void her(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     float alpha,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float>       *A, int_t lda );

// /// @ingroup her
// void her(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     double alpha,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double>       *A, int_t lda );

// // -----------------------------------------------------------------------------
// /// @ingroup her2
// void her2(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     float alpha,
//     float const *x, int_t incx,
//     float const *y, int_t incy,
//     float       *A, int_t lda );

// /// @ingroup her2
// void her2(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     double alpha,
//     double const *x, int_t incx,
//     double const *y, int_t incy,
//     double       *A, int_t lda );

// /// @ingroup her2
// void her2(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float> const *y, int_t incy,
//     std::complex<float>       *A, int_t lda );

// /// @ingroup her2
// void her2(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double> const *y, int_t incy,
//     std::complex<double>       *A, int_t lda );

// // -----------------------------------------------------------------------------
// /// @ingroup symv
// void symv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     float alpha,
//     float const *A, int_t lda,
//     float const *x, int_t incx,
//     float beta,
//     float       *y, int_t incy );

// /// @ingroup symv
// void symv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     double alpha,
//     double const *A, int_t lda,
//     double const *x, int_t incx,
//     double beta,
//     double       *y, int_t incy );

// /// @ingroup symv
// void symv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float> beta,
//     std::complex<float>       *y, int_t incy );

// /// @ingroup symv
// void symv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double> beta,
//     std::complex<double>       *y, int_t incy );

// // -----------------------------------------------------------------------------
// // only real; complex in lapack++
// /// @ingroup syr
// void syr(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     float alpha,
//     float const *x, int_t incx,
//     float       *A, int_t lda );

// /// @ingroup syr
// void syr(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     double alpha,
//     double const *x, int_t incx,
//     double       *A, int_t lda );

// // -----------------------------------------------------------------------------
// /// @ingroup syr2
// void syr2(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     float alpha,
//     float const *x, int_t incx,
//     float const *y, int_t incy,
//     float       *A, int_t lda );

// /// @ingroup syr2
// void syr2(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     double alpha,
//     double const *x, int_t incx,
//     double const *y, int_t incy,
//     double       *A, int_t lda );

// /// @ingroup syr2
// void syr2(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *x, int_t incx,
//     std::complex<float> const *y, int_t incy,
//     std::complex<float>       *A, int_t lda );

// /// @ingroup syr2
// void syr2(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *x, int_t incx,
//     std::complex<double> const *y, int_t incy,
//     std::complex<double>       *A, int_t lda );

// // -----------------------------------------------------------------------------
// /// @ingroup trmv
// void trmv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t n,
//     float const *A, int_t lda,
//     float       *x, int_t incx );

// /// @ingroup trmv
// void trmv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t n,
//     double const *A, int_t lda,
//     double       *x, int_t incx );

// /// @ingroup trmv
// void trmv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t n,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float>       *x, int_t incx );

// /// @ingroup trmv
// void trmv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t n,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double>       *x, int_t incx );

// // -----------------------------------------------------------------------------
// /// @ingroup trsv
// void trsv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t n,
//     float const *A, int_t lda,
//     float       *x, int_t incx );

// /// @ingroup trsv
// void trsv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t n,
//     double const *A, int_t lda,
//     double       *x, int_t incx );

// /// @ingroup trsv
// void trsv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t n,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float>       *x, int_t incx );

// /// @ingroup trsv
// void trsv(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t n,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double>       *x, int_t incx );
// // =============================================================================
// // Level 3 BLAS

// // -----------------------------------------------------------------------------
// #ifndef BLAS_USE_TEMPLATE_GEMM

// /// @ingroup gemm
// void gemm(
//     blas::Layout layout,
//     blas::Op transA,
//     blas::Op transB,
//     tlapack::size_t m, tlapack::size_t n, tlapack::size_t k,
//     float alpha,
//     float const *A, int_t lda,
//     float const *B, int_t ldb,
//     float beta,
//     float       *C, int_t ldc );

// /// @ingroup gemm
// void gemm(
//     blas::Layout layout,
//     blas::Op transA,
//     blas::Op transB,
//     tlapack::size_t m, tlapack::size_t n, tlapack::size_t k,
//     double alpha,
//     double const *A, int_t lda,
//     double const *B, int_t ldb,
//     double beta,
//     double       *C, int_t ldc );

// /// @ingroup gemm
// void gemm(
//     blas::Layout layout,
//     blas::Op transA,
//     blas::Op transB,
//     tlapack::size_t m, tlapack::size_t n, tlapack::size_t k,
//     std::complex<float> alpha,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float> const *B, int_t ldb,
//     std::complex<float> beta,
//     std::complex<float>       *C, int_t ldc );

// /// @ingroup gemm
// void gemm(
//     blas::Layout layout,
//     blas::Op transA,
//     blas::Op transB,
//     tlapack::size_t m, tlapack::size_t n, tlapack::size_t k,
//     std::complex<double> alpha,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double> const *B, int_t ldb,
//     std::complex<double> beta,
//     std::complex<double>       *C, int_t ldc );

// #endif

// // -----------------------------------------------------------------------------
// /// @ingroup hemm
// void hemm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     tlapack::size_t m, tlapack::size_t n,
//     float alpha,
//     float const *A, int_t lda,
//     float const *B, int_t ldb,
//     float beta,
//     float       *C, int_t ldc );

// /// @ingroup hemm
// void hemm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     tlapack::size_t m, tlapack::size_t n,
//     double alpha,
//     double const *A, int_t lda,
//     double const *B, int_t ldb,
//     double beta,
//     double       *C, int_t ldc );

// /// @ingroup hemm
// void hemm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     tlapack::size_t m, tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float> const *B, int_t ldb,
//     std::complex<float> beta,
//     std::complex<float>       *C, int_t ldc );

// /// @ingroup hemm
// void hemm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     tlapack::size_t m, tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double> const *B, int_t ldb,
//     std::complex<double> beta,
//     std::complex<double>       *C, int_t ldc );

// // -----------------------------------------------------------------------------
// /// @ingroup her2k
// void her2k(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     float alpha,
//     float const *A, int_t lda,
//     float const *B, int_t ldb,
//     float beta,
//     float       *C, int_t ldc );

// /// @ingroup her2k
// void her2k(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     double alpha,
//     double const *A, int_t lda,
//     double const *B, int_t ldb,
//     double beta,
//     double       *C, int_t ldc );

// /// @ingroup her2k
// void her2k(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     std::complex<float> alpha,  // note: complex
//     std::complex<float> const *A, int_t lda,
//     std::complex<float> const *B, int_t ldb,
//     float beta,   // note: real
//     std::complex<float>       *C, int_t ldc );

// /// @ingroup her2k
// void her2k(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     std::complex<double> alpha,  // note: complex
//     std::complex<double> const *A, int_t lda,
//     std::complex<double> const *B, int_t ldb,
//     double beta,  // note: real
//     std::complex<double>       *C, int_t ldc );

// // -----------------------------------------------------------------------------
// /// @ingroup herk
// void herk(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     float alpha,
//     float const *A, int_t lda,
//     float beta,
//     float       *C, int_t ldc );

// /// @ingroup herk
// void herk(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     double alpha,
//     double const *A, int_t lda,
//     double beta,
//     double       *C, int_t ldc );

// /// @ingroup herk
// void herk(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     float alpha,  // note: real
//     std::complex<float> const *A, int_t lda,
//     float beta,   // note: real
//     std::complex<float>       *C, int_t ldc );

// /// @ingroup herk
// void herk(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     double alpha,
//     std::complex<double> const *A, int_t lda,
//     double beta,
//     std::complex<double>       *C, int_t ldc );

// // -----------------------------------------------------------------------------
// /// @ingroup symm
// void symm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     tlapack::size_t m, tlapack::size_t n,
//     float alpha,
//     float const *A, int_t lda,
//     float const *B, int_t ldb,
//     float beta,
//     float       *C, int_t ldc );

// /// @ingroup symm
// void symm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     tlapack::size_t m, tlapack::size_t n,
//     double alpha,
//     double const *A, int_t lda,
//     double const *B, int_t ldb,
//     double beta,
//     double       *C, int_t ldc );

// /// @ingroup symm
// void symm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     tlapack::size_t m, tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float> const *B, int_t ldb,
//     std::complex<float> beta,
//     std::complex<float>       *C, int_t ldc );

// /// @ingroup symm
// void symm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     tlapack::size_t m, tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double> const *B, int_t ldb,
//     std::complex<double> beta,
//     std::complex<double>       *C, int_t ldc );

// // -----------------------------------------------------------------------------
// /// @ingroup syr2k
// void syr2k(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     float alpha,
//     float const *A, int_t lda,
//     float const *B, int_t ldb,
//     float beta,
//     float       *C, int_t ldc );

// /// @ingroup syr2k
// void syr2k(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     double alpha,
//     double const *A, int_t lda,
//     double const *B, int_t ldb,
//     double beta,
//     double       *C, int_t ldc );

// /// @ingroup syr2k
// void syr2k(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     std::complex<float> alpha,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float> const *B, int_t ldb,
//     std::complex<float> beta,
//     std::complex<float>       *C, int_t ldc );

// /// @ingroup syr2k
// void syr2k(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     std::complex<double> alpha,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double> const *B, int_t ldb,
//     std::complex<double> beta,
//     std::complex<double>       *C, int_t ldc );

// // -----------------------------------------------------------------------------
// /// @ingroup syrk
// void syrk(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     float alpha,
//     float const *A, int_t lda,
//     float beta,
//     float       *C, int_t ldc );

// /// @ingroup syrk
// void syrk(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     double alpha,
//     double const *A, int_t lda,
//     double beta,
//     double       *C, int_t ldc );

// /// @ingroup syrk
// void syrk(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     std::complex<float> alpha,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float> beta,
//     std::complex<float>       *C, int_t ldc );

// /// @ingroup syrk
// void syrk(
//     blas::Layout layout,
//     blas::Uplo uplo,
//     blas::Op trans,
//     tlapack::size_t n, tlapack::size_t k,
//     std::complex<double> alpha,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double> beta,
//     std::complex<double>       *C, int_t ldc );

// // -----------------------------------------------------------------------------
// /// @ingroup trmm
// void trmm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t m,
//     tlapack::size_t n,
//     float alpha,
//     float const *A, int_t lda,
//     float       *B, int_t ldb );

// /// @ingroup trmm
// void trmm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t m,
//     tlapack::size_t n,
//     double alpha,
//     double const *A, int_t lda,
//     double       *B, int_t ldb );

// /// @ingroup trmm
// void trmm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t m,
//     tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float>       *B, int_t ldb );

// /// @ingroup trmm
// void trmm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t m,
//     tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double>       *B, int_t ldb );

// // -----------------------------------------------------------------------------
// /// @ingroup trsm
// void trsm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t m,
//     tlapack::size_t n,
//     float alpha,
//     float const *A, int_t lda,
//     float       *B, int_t ldb );

// /// @ingroup trsm
// void trsm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t m,
//     tlapack::size_t n,
//     double alpha,
//     double const *A, int_t lda,
//     double       *B, int_t ldb );

// /// @ingroup trsm
// void trsm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t m,
//     tlapack::size_t n,
//     std::complex<float> alpha,
//     std::complex<float> const *A, int_t lda,
//     std::complex<float>       *B, int_t ldb );

// /// @ingroup trsm
// void trsm(
//     blas::Layout layout,
//     blas::Side side,
//     blas::Uplo uplo,
//     blas::Op trans,
//     blas::Diag diag,
//     tlapack::size_t m,
//     tlapack::size_t n,
//     std::complex<double> alpha,
//     std::complex<double> const *A, int_t lda,
//     std::complex<double>       *B, int_t ldb );

// }