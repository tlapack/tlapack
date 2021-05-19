#ifndef __BLAS_H__
#define __BLAS_H__

// -----------------------------------------------------------------------------
#include "defines.h"
#include <stdint.h>
#include <stddef.h>

// -----------------------------------------------------------------------------
// Integer types blas_size_t and blas_int_t
#ifndef BLAS_SIZE_T
    typedef size_t blas_size_t;
#else
    typedef BLAS_SIZE_T blas_size_t;
#endif

#ifndef BLAS_INT_T
    typedef int64_t blas_int_t;
#else
    typedef BLAS_INT_T blas_int_t;
#endif

// -----------------------------------------------------------------------------
// Complex types
#include <complex.h>
typedef float  _Complex complexFloat;
typedef double _Complex complexDouble;

// -----------------------------------------------------------------------------
// Other type definitions
typedef enum Layout { ColMajor = 'C', RowMajor = 'R' } Layout;
typedef enum Op     { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' } Op;
typedef enum Uplo   { Upper    = 'U', Lower    = 'L', General   = 'G' } Uplo;
typedef enum Diag   { NonUnit  = 'N', Unit     = 'U' } Diag;
typedef enum Side   { Left     = 'L', Right    = 'R' } Side;

// // -----------------------------------------------------------------------------
// // Mangling
// #if defined(ADD_cblas_PREFIX) && !defined(ADD_)
//     #define BLAS_FUNCTION(fname) cblas_##fname
// #elif !defined(ADD_cblas_PREFIX) && defined(ADD_)
//     #define BLAS_FUNCTION(fname) fname##_
// #elif defined(ADD_cblas_PREFIX) && defined(ADD_)
//     #define BLAS_FUNCTION(fname) cblas_##fname##_
// #else
//     #define BLAS_FUNCTION(fname) fname
// #endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Level 1 BLAS

float sasum(
    blas_size_t n,
    float const * x, blas_int_t incx );

double dasum(
    blas_size_t n,
    double const * x, blas_int_t incx );

float casum(
    blas_size_t n,
    complexFloat const * x, blas_int_t incx );

double zasum(
    blas_size_t n,
    complexDouble const * x, blas_int_t incx );

void saxpy(
    blas_size_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float * y, blas_int_t incy );

void daxpy(
    blas_size_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double * y, blas_int_t incy );

void caxpy(
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat * y, blas_int_t incy );

void zaxpy(
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble * y, blas_int_t incy );

void scopy(
    blas_size_t n,
    float const * x, blas_int_t incx,
    float * y, blas_int_t incy );

void dcopy(
    blas_size_t n,
    double const * x, blas_int_t incx,
    double * y, blas_int_t incy );

void ccopy(
    blas_size_t n,
    complexFloat const * x, blas_int_t incx,
    complexFloat * y, blas_int_t incy );

void zcopy(
    blas_size_t n,
    complexDouble const * x, blas_int_t incx,
    complexDouble * y, blas_int_t incy );

float sdot(
    blas_size_t n,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy );

double ddot(
    blas_size_t n,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy );

complexFloat cdot(
    blas_size_t n,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy );

complexDouble zdot(
    blas_size_t n,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy );

float sdotu(
    blas_size_t n,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy );

double ddotu(
    blas_size_t n,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy );

complexFloat cdotu(
    blas_size_t n,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy );

complexDouble zdotu(
    blas_size_t n,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy );

blas_size_t siamax(
    blas_size_t n,
    float const * x, blas_int_t incx );

blas_size_t diamax(
    blas_size_t n,
    double const * x, blas_int_t incx );

blas_size_t ciamax(
    blas_size_t n,
    complexFloat const * x, blas_int_t incx );

blas_size_t ziamax(
    blas_size_t n,
    complexDouble const * x, blas_int_t incx );

float snrm2(
    blas_size_t n,
    float const * x, blas_int_t incx );

double dnrm2(
    blas_size_t n,
    double const * x, blas_int_t incx );

float cnrm2(
    blas_size_t n,
    complexFloat const * x, blas_int_t incx );

double znrm2(
    blas_size_t n,
    complexDouble const * x, blas_int_t incx );

void srot(
    blas_size_t n,
    float * x, blas_int_t incx,
    float * y, blas_int_t incy,
    float c,
    float s );

void drot(
    blas_size_t n,
    double * x, blas_int_t incx,
    double * y, blas_int_t incy,
    double c,
    double s );

void csrot(
    blas_size_t n,
    complexFloat * x, blas_int_t incx,
    complexFloat * y, blas_int_t incy,
    float c,
    float s );

void zdrot(
    blas_size_t n,
    complexDouble * x, blas_int_t incx,
    complexDouble * y, blas_int_t incy,
    double c,
    double s );

void crot(
    blas_size_t n,
    complexFloat * x, blas_int_t incx,
    complexFloat * y, blas_int_t incy,
    float c,
    complexFloat s );

void zrot(
    blas_size_t n,
    complexDouble * x, blas_int_t incx,
    complexDouble * y, blas_int_t incy,
    double c,
    complexDouble s );

void srotg(
    float * a,
    float * b,
    float * c,
    float * s );

void drotg(
    double * a,
    double * b,
    double * c,
    double * s );

void crotg(
    complexFloat * a,
    complexFloat * b,
    float * c,
    complexFloat * s );

void zrotg(
    complexDouble * a,
    complexDouble * b,
    double * c,
    complexDouble * s );

void srotm(
    blas_size_t n,
    float * x, blas_int_t incx,
    float * y, blas_int_t incy,
    float const * param );

void drotm(
    blas_size_t n,
    double * x, blas_int_t incx,
    double * y, blas_int_t incy,
    double const * param );

void srotmg(
    float * d1,
    float * d2,
    float * a,
    float b,
    float * param );

void drotmg(
    double * d1,
    double * d2,
    double * a,
    double b,
    double * param );

void sscal(
    blas_size_t n,
    float alpha,
    float * x, blas_int_t incx );

void dscal(
    blas_size_t n,
    double alpha,
    double * x, blas_int_t incx );

void cscal(
    blas_size_t n,
    complexFloat alpha,
    complexFloat * x, blas_int_t incx );

void zscal(
    blas_size_t n,
    complexDouble alpha,
    complexDouble * x, blas_int_t incx );

void sswap(
    blas_size_t n,
    float * x, blas_int_t incx,
    float * y, blas_int_t incy );

void dswap(
    blas_size_t n,
    double * x, blas_int_t incx,
    double * y, blas_int_t incy );

void cswap(
    blas_size_t n,
    complexFloat * x, blas_int_t incx,
    complexFloat * y, blas_int_t incy );

void zswap(
    blas_size_t n,
    complexDouble * x, blas_int_t incx,
    complexDouble * y, blas_int_t incy );

// =============================================================================
// Level 2 BLAS

void sgemv(
    Layout layout,
    Op trans,
    blas_size_t m,
    blas_size_t n,
    float alpha,
    float const * A, blas_size_t lda,
    float const * x, blas_int_t incx,
    float beta,
    float * y, blas_int_t incy );

void dgemv(
    Layout layout,
    Op trans,
    blas_size_t m,
    blas_size_t n,
    double alpha,
    double const * A, blas_size_t lda,
    double const * x, blas_int_t incx,
    double beta,
    double * y, blas_int_t incy );

void cgemv(
    Layout layout,
    Op trans,
    blas_size_t m,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat const * x, blas_int_t incx,
    complexFloat beta,
    complexFloat * y, blas_int_t incy );

void zgemv(
    Layout layout,
    Op trans,
    blas_size_t m,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble const * x, blas_int_t incx,
    complexDouble beta,
    complexDouble * y, blas_int_t incy );

void sger(
    Layout layout,
    blas_size_t m,
    blas_size_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy,
    float * A, blas_size_t lda );

void dger(
    Layout layout,
    blas_size_t m,
    blas_size_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy,
    double * A, blas_size_t lda );

void cger(
    Layout layout,
    blas_size_t m,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy,
    complexFloat * A, blas_size_t lda );

void zger(
    Layout layout,
    blas_size_t m,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy,
    complexDouble * A, blas_size_t lda );

void sgeru(
    Layout layout,
    blas_size_t m,
    blas_size_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy,
    float * A, blas_size_t lda );

void dgeru(
    Layout layout,
    blas_size_t m,
    blas_size_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy,
    double * A, blas_size_t lda );

void cgeru(
    Layout layout,
    blas_size_t m,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy,
    complexFloat * A, blas_size_t lda );

void zgeru(
    Layout layout,
    blas_size_t m,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy,
    complexDouble * A, blas_size_t lda );

void shemv(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    float alpha,
    float const * A, blas_size_t lda,
    float const * x, blas_int_t incx,
    float beta,
    float * y, blas_int_t incy );

void dhemv(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    double alpha,
    double const * A, blas_size_t lda,
    double const * x, blas_int_t incx,
    double beta,
    double * y, blas_int_t incy );

void chemv(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat const * x, blas_int_t incx,
    complexFloat beta,
    complexFloat * y, blas_int_t incy );

void zhemv(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble const * x, blas_int_t incx,
    complexDouble beta,
    complexDouble * y, blas_int_t incy );

void sher(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float * A, blas_size_t lda );

void dher(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double * A, blas_size_t lda );

void cher(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    float alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat * A, blas_size_t lda );

void zher(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    double alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble * A, blas_size_t lda );

void sher2(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy,
    float * A, blas_size_t lda );

void dher2(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy,
    double * A, blas_size_t lda );

void cher2(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy,
    complexFloat * A, blas_size_t lda );

void zher2(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy,
    complexDouble * A, blas_size_t lda );

void ssymv(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    float alpha,
    float const * A, blas_size_t lda,
    float const * x, blas_int_t incx,
    float beta,
    float * y, blas_int_t incy );

void dsymv(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    double alpha,
    double const * A, blas_size_t lda,
    double const * x, blas_int_t incx,
    double beta,
    double * y, blas_int_t incy );

void csymv(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat const * x, blas_int_t incx,
    complexFloat beta,
    complexFloat * y, blas_int_t incy );

void zsymv(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble const * x, blas_int_t incx,
    complexDouble beta,
    complexDouble * y, blas_int_t incy );

void ssyr(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float * A, blas_size_t lda );

void dsyr(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double * A, blas_size_t lda );

void ssyr2(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy,
    float * A, blas_size_t lda );

void dsyr2(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy,
    double * A, blas_size_t lda );

void csyr2(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy,
    complexFloat * A, blas_size_t lda );

void zsyr2(
    Layout layout,
    Uplo uplo,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy,
    complexDouble * A, blas_size_t lda );

void strmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t n,
    float const * A, blas_size_t lda,
    float * x, blas_int_t incx );

void dtrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t n,
    double const * A, blas_size_t lda,
    double * x, blas_int_t incx );

void ctrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t n,
    complexFloat const * A, blas_size_t lda,
    complexFloat * x, blas_int_t incx );

void ztrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t n,
    complexDouble const * A, blas_size_t lda,
    complexDouble * x, blas_int_t incx );

void strsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t n,
    float const * A, blas_size_t lda,
    float * x, blas_int_t incx );

void dtrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t n,
    double const * A, blas_size_t lda,
    double * x, blas_int_t incx );

void ctrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t n,
    complexFloat const * A, blas_size_t lda,
    complexFloat * x, blas_int_t incx );

void ztrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t n,
    complexDouble const * A, blas_size_t lda,
    complexDouble * x, blas_int_t incx );

// =============================================================================
// Level 3 BLAS

void sgemm(
    Layout layout,
    Op transA,
    Op transB,
    blas_size_t m,
    blas_size_t n,
    blas_size_t k,
    float alpha,
    float const * A, blas_size_t lda,
    float const * B, blas_size_t ldb,
    float beta,
    float * C, blas_size_t ldc );

void dgemm(
    Layout layout,
    Op transA,
    Op transB,
    blas_size_t m,
    blas_size_t n,
    blas_size_t k,
    double alpha,
    double const * A, blas_size_t lda,
    double const * B, blas_size_t ldb,
    double beta,
    double * C, blas_size_t ldc );

void cgemm(
    Layout layout,
    Op transA,
    Op transB,
    blas_size_t m,
    blas_size_t n,
    blas_size_t k,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat const * B, blas_size_t ldb,
    complexFloat beta,
    complexFloat * C, blas_size_t ldc );

void zgemm(
    Layout layout,
    Op transA,
    Op transB,
    blas_size_t m,
    blas_size_t n,
    blas_size_t k,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble const * B, blas_size_t ldb,
    complexDouble beta,
    complexDouble * C, blas_size_t ldc );

void shemm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_size_t m,
    blas_size_t n,
    float alpha,
    float const * A, blas_size_t lda,
    float const * B, blas_size_t ldb,
    float beta,
    float * C, blas_size_t ldc );

void dhemm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_size_t m,
    blas_size_t n,
    double alpha,
    double const * A, blas_size_t lda,
    double const * B, blas_size_t ldb,
    double beta,
    double * C, blas_size_t ldc );

void chemm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_size_t m,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat const * B, blas_size_t ldb,
    complexFloat beta,
    complexFloat * C, blas_size_t ldc );

void zhemm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_size_t m,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble const * B, blas_size_t ldb,
    complexDouble beta,
    complexDouble * C, blas_size_t ldc );

void sher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    float alpha,
    float const * A, blas_size_t lda,
    float const * B, blas_size_t ldb,
    float beta,
    float * C, blas_size_t ldc );

void dher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    double alpha,
    double const * A, blas_size_t lda,
    double const * B, blas_size_t ldb,
    double beta,
    double * C, blas_size_t ldc );

void cher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat const * B, blas_size_t ldb,
    float beta,
    complexFloat * C, blas_size_t ldc );

void zher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble const * B, blas_size_t ldb,
    double beta,
    complexDouble * C, blas_size_t ldc );

void sherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    float alpha,
    float const * A, blas_size_t lda,
    float beta,
    float * C, blas_size_t ldc );

void dherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    double alpha,
    double const * A, blas_size_t lda,
    double beta,
    double * C, blas_size_t ldc );

void cherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    float alpha,
    complexFloat const * A, blas_size_t lda,
    float beta,
    complexFloat * C, blas_size_t ldc );

void zherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    double alpha,
    complexDouble const * A, blas_size_t lda,
    double beta,
    complexDouble * C, blas_size_t ldc );

void ssymm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_size_t m,
    blas_size_t n,
    float alpha,
    float const * A, blas_size_t lda,
    float const * B, blas_size_t ldb,
    float beta,
    float * C, blas_size_t ldc );

void dsymm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_size_t m,
    blas_size_t n,
    double alpha,
    double const * A, blas_size_t lda,
    double const * B, blas_size_t ldb,
    double beta,
    double * C, blas_size_t ldc );

void csymm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_size_t m,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat const * B, blas_size_t ldb,
    complexFloat beta,
    complexFloat * C, blas_size_t ldc );

void zsymm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_size_t m,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble const * B, blas_size_t ldb,
    complexDouble beta,
    complexDouble * C, blas_size_t ldc );

void ssyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    float alpha,
    float const * A, blas_size_t lda,
    float const * B, blas_size_t ldb,
    float beta,
    float * C, blas_size_t ldc );

void dsyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    double alpha,
    double const * A, blas_size_t lda,
    double const * B, blas_size_t ldb,
    double beta,
    double * C, blas_size_t ldc );

void csyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat const * B, blas_size_t ldb,
    complexFloat beta,
    complexFloat * C, blas_size_t ldc );

void zsyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble const * B, blas_size_t ldb,
    complexDouble beta,
    complexDouble * C, blas_size_t ldc );

void ssyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    float alpha,
    float const * A, blas_size_t lda,
    float beta,
    float * C, blas_size_t ldc );

void dsyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    double alpha,
    double const * A, blas_size_t lda,
    double beta,
    double * C, blas_size_t ldc );

void csyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat beta,
    complexFloat * C, blas_size_t ldc );

void zsyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_size_t n,
    blas_size_t k,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble beta,
    complexDouble * C, blas_size_t ldc );

void strmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t m,
    blas_size_t n,
    float alpha,
    float const * A, blas_size_t lda,
    float * B, blas_size_t ldb );

void dtrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t m,
    blas_size_t n,
    double alpha,
    double const * A, blas_size_t lda,
    double * B, blas_size_t ldb );

void ctrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t m,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat * B, blas_size_t ldb );

void ztrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t m,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble * B, blas_size_t ldb );

void strsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t m,
    blas_size_t n,
    float alpha,
    float const * A, blas_size_t lda,
    float * B, blas_size_t ldb );

void dtrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t m,
    blas_size_t n,
    double alpha,
    double const * A, blas_size_t lda,
    double * B, blas_size_t ldb );

void ctrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t m,
    blas_size_t n,
    complexFloat alpha,
    complexFloat const * A, blas_size_t lda,
    complexFloat * B, blas_size_t ldb );

void ztrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_size_t m,
    blas_size_t n,
    complexDouble alpha,
    complexDouble const * A, blas_size_t lda,
    complexDouble * B, blas_size_t ldb );

#ifdef __cplusplus
}
#endif

#endif // __BLAS_H__