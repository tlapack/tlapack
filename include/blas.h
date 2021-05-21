#ifndef __BLAS_H__
#define __BLAS_H__

// -----------------------------------------------------------------------------
#include "defines.h"
#include <stdint.h>
#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Integer types BLAS_SIZE_T and BLAS_INT_T
#ifndef BLAS_SIZE_T
    #define BLAS_SIZE_T size_t
#endif
#ifndef BLAS_INT_T
    #define BLAS_SIZE_T int64_t
#endif

// -----------------------------------------------------------------------------
// Enumerations
typedef enum Layout { ColMajor = 'C', RowMajor = 'R' } Layout;
typedef enum Op     { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' } Op;
typedef enum Uplo   { Upper    = 'U', Lower    = 'L', General   = 'G' } Uplo;
typedef enum Diag   { NonUnit  = 'N', Unit     = 'U' } Diag;
typedef enum Side   { Left     = 'L', Right    = 'R' } Side;

// =============================================================================
// Level 1 BLAS

float sasum(
    BLAS_SIZE_T n,
    float const * x, BLAS_INT_T incx );

double dasum(
    BLAS_SIZE_T n,
    double const * x, BLAS_INT_T incx );

float casum(
    BLAS_SIZE_T n,
    float  _Complex const * x, BLAS_INT_T incx );

double zasum(
    BLAS_SIZE_T n,
    double _Complex const * x, BLAS_INT_T incx );

void saxpy(
    BLAS_SIZE_T n,
    float alpha,
    float const * x, BLAS_INT_T incx,
    float * y, BLAS_INT_T incy );

void daxpy(
    BLAS_SIZE_T n,
    double alpha,
    double const * x, BLAS_INT_T incx,
    double * y, BLAS_INT_T incy );

void caxpy(
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex * y, BLAS_INT_T incy );

void zaxpy(
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex * y, BLAS_INT_T incy );

void scopy(
    BLAS_SIZE_T n,
    float const * x, BLAS_INT_T incx,
    float * y, BLAS_INT_T incy );

void dcopy(
    BLAS_SIZE_T n,
    double const * x, BLAS_INT_T incx,
    double * y, BLAS_INT_T incy );

void ccopy(
    BLAS_SIZE_T n,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex * y, BLAS_INT_T incy );

void zcopy(
    BLAS_SIZE_T n,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex * y, BLAS_INT_T incy );

float sdot(
    BLAS_SIZE_T n,
    float const * x, BLAS_INT_T incx,
    float const * y, BLAS_INT_T incy );

double ddot(
    BLAS_SIZE_T n,
    double const * x, BLAS_INT_T incx,
    double const * y, BLAS_INT_T incy );

float  _Complex cdot(
    BLAS_SIZE_T n,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex const * y, BLAS_INT_T incy );

double _Complex zdot(
    BLAS_SIZE_T n,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex const * y, BLAS_INT_T incy );

float sdotu(
    BLAS_SIZE_T n,
    float const * x, BLAS_INT_T incx,
    float const * y, BLAS_INT_T incy );

double ddotu(
    BLAS_SIZE_T n,
    double const * x, BLAS_INT_T incx,
    double const * y, BLAS_INT_T incy );

float  _Complex cdotu(
    BLAS_SIZE_T n,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex const * y, BLAS_INT_T incy );

double _Complex zdotu(
    BLAS_SIZE_T n,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex const * y, BLAS_INT_T incy );

BLAS_SIZE_T siamax(
    BLAS_SIZE_T n,
    float const * x, BLAS_INT_T incx );

BLAS_SIZE_T diamax(
    BLAS_SIZE_T n,
    double const * x, BLAS_INT_T incx );

BLAS_SIZE_T ciamax(
    BLAS_SIZE_T n,
    float  _Complex const * x, BLAS_INT_T incx );

BLAS_SIZE_T ziamax(
    BLAS_SIZE_T n,
    double _Complex const * x, BLAS_INT_T incx );

float snrm2(
    BLAS_SIZE_T n,
    float const * x, BLAS_INT_T incx );

double dnrm2(
    BLAS_SIZE_T n,
    double const * x, BLAS_INT_T incx );

float cnrm2(
    BLAS_SIZE_T n,
    float  _Complex const * x, BLAS_INT_T incx );

double znrm2(
    BLAS_SIZE_T n,
    double _Complex const * x, BLAS_INT_T incx );

void srot(
    BLAS_SIZE_T n,
    float * x, BLAS_INT_T incx,
    float * y, BLAS_INT_T incy,
    float c,
    float s );

void drot(
    BLAS_SIZE_T n,
    double * x, BLAS_INT_T incx,
    double * y, BLAS_INT_T incy,
    double c,
    double s );

void csrot(
    BLAS_SIZE_T n,
    float  _Complex * x, BLAS_INT_T incx,
    float  _Complex * y, BLAS_INT_T incy,
    float c,
    float s );

void zdrot(
    BLAS_SIZE_T n,
    double _Complex * x, BLAS_INT_T incx,
    double _Complex * y, BLAS_INT_T incy,
    double c,
    double s );

void crot(
    BLAS_SIZE_T n,
    float  _Complex * x, BLAS_INT_T incx,
    float  _Complex * y, BLAS_INT_T incy,
    float c,
    float  _Complex s );

void zrot(
    BLAS_SIZE_T n,
    double _Complex * x, BLAS_INT_T incx,
    double _Complex * y, BLAS_INT_T incy,
    double c,
    double _Complex s );

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
    float  _Complex * a,
    float  _Complex * b,
    float * c,
    float  _Complex * s );

void zrotg(
    double _Complex * a,
    double _Complex * b,
    double * c,
    double _Complex * s );

void srotm(
    BLAS_SIZE_T n,
    float * x, BLAS_INT_T incx,
    float * y, BLAS_INT_T incy,
    float const * param );

void drotm(
    BLAS_SIZE_T n,
    double * x, BLAS_INT_T incx,
    double * y, BLAS_INT_T incy,
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
    BLAS_SIZE_T n,
    float alpha,
    float * x, BLAS_INT_T incx );

void dscal(
    BLAS_SIZE_T n,
    double alpha,
    double * x, BLAS_INT_T incx );

void cscal(
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex * x, BLAS_INT_T incx );

void zscal(
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex * x, BLAS_INT_T incx );

void sswap(
    BLAS_SIZE_T n,
    float * x, BLAS_INT_T incx,
    float * y, BLAS_INT_T incy );

void dswap(
    BLAS_SIZE_T n,
    double * x, BLAS_INT_T incx,
    double * y, BLAS_INT_T incy );

void cswap(
    BLAS_SIZE_T n,
    float  _Complex * x, BLAS_INT_T incx,
    float  _Complex * y, BLAS_INT_T incy );

void zswap(
    BLAS_SIZE_T n,
    double _Complex * x, BLAS_INT_T incx,
    double _Complex * y, BLAS_INT_T incy );

// =============================================================================
// Level 2 BLAS

void sgemv(
    Layout layout,
    Op trans,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float const * x, BLAS_INT_T incx,
    float beta,
    float * y, BLAS_INT_T incy );

void dgemv(
    Layout layout,
    Op trans,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double const * x, BLAS_INT_T incx,
    double beta,
    double * y, BLAS_INT_T incy );

void cgemv(
    Layout layout,
    Op trans,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex beta,
    float  _Complex * y, BLAS_INT_T incy );

void zgemv(
    Layout layout,
    Op trans,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex beta,
    double _Complex * y, BLAS_INT_T incy );

void sger(
    Layout layout,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float alpha,
    float const * x, BLAS_INT_T incx,
    float const * y, BLAS_INT_T incy,
    float * A, BLAS_SIZE_T lda );

void dger(
    Layout layout,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double alpha,
    double const * x, BLAS_INT_T incx,
    double const * y, BLAS_INT_T incy,
    double * A, BLAS_SIZE_T lda );

void cger(
    Layout layout,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex const * y, BLAS_INT_T incy,
    float  _Complex * A, BLAS_SIZE_T lda );

void zger(
    Layout layout,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex const * y, BLAS_INT_T incy,
    double _Complex * A, BLAS_SIZE_T lda );

void sgeru(
    Layout layout,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float alpha,
    float const * x, BLAS_INT_T incx,
    float const * y, BLAS_INT_T incy,
    float * A, BLAS_SIZE_T lda );

void dgeru(
    Layout layout,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double alpha,
    double const * x, BLAS_INT_T incx,
    double const * y, BLAS_INT_T incy,
    double * A, BLAS_SIZE_T lda );

void cgeru(
    Layout layout,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex const * y, BLAS_INT_T incy,
    float  _Complex * A, BLAS_SIZE_T lda );

void zgeru(
    Layout layout,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex const * y, BLAS_INT_T incy,
    double _Complex * A, BLAS_SIZE_T lda );

void shemv(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float const * x, BLAS_INT_T incx,
    float beta,
    float * y, BLAS_INT_T incy );

void dhemv(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double const * x, BLAS_INT_T incx,
    double beta,
    double * y, BLAS_INT_T incy );

void chemv(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex beta,
    float  _Complex * y, BLAS_INT_T incy );

void zhemv(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex beta,
    double _Complex * y, BLAS_INT_T incy );

void sher(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float alpha,
    float const * x, BLAS_INT_T incx,
    float * A, BLAS_SIZE_T lda );

void dher(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double alpha,
    double const * x, BLAS_INT_T incx,
    double * A, BLAS_SIZE_T lda );

void cher(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float alpha,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex * A, BLAS_SIZE_T lda );

void zher(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double alpha,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex * A, BLAS_SIZE_T lda );

void sher2(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float alpha,
    float const * x, BLAS_INT_T incx,
    float const * y, BLAS_INT_T incy,
    float * A, BLAS_SIZE_T lda );

void dher2(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double alpha,
    double const * x, BLAS_INT_T incx,
    double const * y, BLAS_INT_T incy,
    double * A, BLAS_SIZE_T lda );

void cher2(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex const * y, BLAS_INT_T incy,
    float  _Complex * A, BLAS_SIZE_T lda );

void zher2(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex const * y, BLAS_INT_T incy,
    double _Complex * A, BLAS_SIZE_T lda );

void ssymv(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float const * x, BLAS_INT_T incx,
    float beta,
    float * y, BLAS_INT_T incy );

void dsymv(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double const * x, BLAS_INT_T incx,
    double beta,
    double * y, BLAS_INT_T incy );

void csymv(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex beta,
    float  _Complex * y, BLAS_INT_T incy );

void zsymv(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex beta,
    double _Complex * y, BLAS_INT_T incy );

void ssyr(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float alpha,
    float const * x, BLAS_INT_T incx,
    float * A, BLAS_SIZE_T lda );

void dsyr(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double alpha,
    double const * x, BLAS_INT_T incx,
    double * A, BLAS_SIZE_T lda );

void ssyr2(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float alpha,
    float const * x, BLAS_INT_T incx,
    float const * y, BLAS_INT_T incy,
    float * A, BLAS_SIZE_T lda );

void dsyr2(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double alpha,
    double const * x, BLAS_INT_T incx,
    double const * y, BLAS_INT_T incy,
    double * A, BLAS_SIZE_T lda );

void csyr2(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * x, BLAS_INT_T incx,
    float  _Complex const * y, BLAS_INT_T incy,
    float  _Complex * A, BLAS_SIZE_T lda );

void zsyr2(
    Layout layout,
    Uplo uplo,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * x, BLAS_INT_T incx,
    double _Complex const * y, BLAS_INT_T incy,
    double _Complex * A, BLAS_SIZE_T lda );

void strmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T n,
    float const * A, BLAS_SIZE_T lda,
    float * x, BLAS_INT_T incx );

void dtrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T n,
    double const * A, BLAS_SIZE_T lda,
    double * x, BLAS_INT_T incx );

void ctrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T n,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex * x, BLAS_INT_T incx );

void ztrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T n,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex * x, BLAS_INT_T incx );

void strsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T n,
    float const * A, BLAS_SIZE_T lda,
    float * x, BLAS_INT_T incx );

void dtrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T n,
    double const * A, BLAS_SIZE_T lda,
    double * x, BLAS_INT_T incx );

void ctrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T n,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex * x, BLAS_INT_T incx );

void ztrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T n,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex * x, BLAS_INT_T incx );

// =============================================================================
// Level 3 BLAS

void sgemm(
    Layout layout,
    Op transA,
    Op transB,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float const * B, BLAS_SIZE_T ldb,
    float beta,
    float * C, BLAS_SIZE_T ldc );

void dgemm(
    Layout layout,
    Op transA,
    Op transB,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double const * B, BLAS_SIZE_T ldb,
    double beta,
    double * C, BLAS_SIZE_T ldc );

void cgemm(
    Layout layout,
    Op transA,
    Op transB,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex const * B, BLAS_SIZE_T ldb,
    float  _Complex beta,
    float  _Complex * C, BLAS_SIZE_T ldc );

void zgemm(
    Layout layout,
    Op transA,
    Op transB,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex const * B, BLAS_SIZE_T ldb,
    double _Complex beta,
    double _Complex * C, BLAS_SIZE_T ldc );

void shemm(
    Layout layout,
    Side side,
    Uplo uplo,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float const * B, BLAS_SIZE_T ldb,
    float beta,
    float * C, BLAS_SIZE_T ldc );

void dhemm(
    Layout layout,
    Side side,
    Uplo uplo,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double const * B, BLAS_SIZE_T ldb,
    double beta,
    double * C, BLAS_SIZE_T ldc );

void chemm(
    Layout layout,
    Side side,
    Uplo uplo,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex const * B, BLAS_SIZE_T ldb,
    float  _Complex beta,
    float  _Complex * C, BLAS_SIZE_T ldc );

void zhemm(
    Layout layout,
    Side side,
    Uplo uplo,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex const * B, BLAS_SIZE_T ldb,
    double _Complex beta,
    double _Complex * C, BLAS_SIZE_T ldc );

void sher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float const * B, BLAS_SIZE_T ldb,
    float beta,
    float * C, BLAS_SIZE_T ldc );

void dher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double const * B, BLAS_SIZE_T ldb,
    double beta,
    double * C, BLAS_SIZE_T ldc );

void cher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex const * B, BLAS_SIZE_T ldb,
    float beta,
    float  _Complex * C, BLAS_SIZE_T ldc );

void zher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex const * B, BLAS_SIZE_T ldb,
    double beta,
    double _Complex * C, BLAS_SIZE_T ldc );

void sherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float beta,
    float * C, BLAS_SIZE_T ldc );

void dherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double beta,
    double * C, BLAS_SIZE_T ldc );

void cherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    float alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float beta,
    float  _Complex * C, BLAS_SIZE_T ldc );

void zherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    double alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double beta,
    double _Complex * C, BLAS_SIZE_T ldc );

void ssymm(
    Layout layout,
    Side side,
    Uplo uplo,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float const * B, BLAS_SIZE_T ldb,
    float beta,
    float * C, BLAS_SIZE_T ldc );

void dsymm(
    Layout layout,
    Side side,
    Uplo uplo,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double const * B, BLAS_SIZE_T ldb,
    double beta,
    double * C, BLAS_SIZE_T ldc );

void csymm(
    Layout layout,
    Side side,
    Uplo uplo,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex const * B, BLAS_SIZE_T ldb,
    float  _Complex beta,
    float  _Complex * C, BLAS_SIZE_T ldc );

void zsymm(
    Layout layout,
    Side side,
    Uplo uplo,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex const * B, BLAS_SIZE_T ldb,
    double _Complex beta,
    double _Complex * C, BLAS_SIZE_T ldc );

void ssyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float const * B, BLAS_SIZE_T ldb,
    float beta,
    float * C, BLAS_SIZE_T ldc );

void dsyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double const * B, BLAS_SIZE_T ldb,
    double beta,
    double * C, BLAS_SIZE_T ldc );

void csyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex const * B, BLAS_SIZE_T ldb,
    float  _Complex beta,
    float  _Complex * C, BLAS_SIZE_T ldc );

void zsyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex const * B, BLAS_SIZE_T ldb,
    double _Complex beta,
    double _Complex * C, BLAS_SIZE_T ldc );

void ssyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float beta,
    float * C, BLAS_SIZE_T ldc );

void dsyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double beta,
    double * C, BLAS_SIZE_T ldc );

void csyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex beta,
    float  _Complex * C, BLAS_SIZE_T ldc );

void zsyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    BLAS_SIZE_T n,
    BLAS_SIZE_T k,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex beta,
    double _Complex * C, BLAS_SIZE_T ldc );

void strmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float * B, BLAS_SIZE_T ldb );

void dtrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double * B, BLAS_SIZE_T ldb );

void ctrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex * B, BLAS_SIZE_T ldb );

void ztrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex * B, BLAS_SIZE_T ldb );

void strsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float alpha,
    float const * A, BLAS_SIZE_T lda,
    float * B, BLAS_SIZE_T ldb );

void dtrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double alpha,
    double const * A, BLAS_SIZE_T lda,
    double * B, BLAS_SIZE_T ldb );

void ctrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    float  _Complex alpha,
    float  _Complex const * A, BLAS_SIZE_T lda,
    float  _Complex * B, BLAS_SIZE_T ldb );

void ztrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    BLAS_SIZE_T m,
    BLAS_SIZE_T n,
    double _Complex alpha,
    double _Complex const * A, BLAS_SIZE_T lda,
    double _Complex * B, BLAS_SIZE_T ldb );

#ifdef __cplusplus
}
#endif

#endif // __BLAS_H__