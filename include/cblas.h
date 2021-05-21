#ifndef __CBLAS_H__
#define __CBLAS_H__

// -----------------------------------------------------------------------------
#include "defines.h"
#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Integer type CBLAS_INDEX
#ifdef WeirdNEC
    #define CBLAS_INDEX long
#else
    #define CBLAS_INDEX int
#endif

// -----------------------------------------------------------------------------
// Enumerations
typedef enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE {CblasLeft=141, CblasRight=142} CBLAS_SIDE;

#define CBLAS_ORDER CBLAS_LAYOUT // this for backward compatibility with CBLAS_ORDER

// =============================================================================
// Level 1 BLAS

float cblas_sasum(
    CBLAS_INDEX n,
    float const * x, CBLAS_INDEX incx );

double cblas_dasum(
    CBLAS_INDEX n,
    double const * x, CBLAS_INDEX incx );

float cblas_casum(
    CBLAS_INDEX n,
    float  _Complex const * x, CBLAS_INDEX incx );

double cblas_zasum(
    CBLAS_INDEX n,
    double _Complex const * x, CBLAS_INDEX incx );

void cblas_saxpy(
    CBLAS_INDEX n,
    float alpha,
    float const * x, CBLAS_INDEX incx,
    float * y, CBLAS_INDEX incy );

void cblas_daxpy(
    CBLAS_INDEX n,
    double alpha,
    double const * x, CBLAS_INDEX incx,
    double * y, CBLAS_INDEX incy );

void cblas_caxpy(
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex * y, CBLAS_INDEX incy );

void cblas_zaxpy(
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex * y, CBLAS_INDEX incy );

void cblas_scopy(
    CBLAS_INDEX n,
    float const * x, CBLAS_INDEX incx,
    float * y, CBLAS_INDEX incy );

void cblas_dcopy(
    CBLAS_INDEX n,
    double const * x, CBLAS_INDEX incx,
    double * y, CBLAS_INDEX incy );

void cblas_ccopy(
    CBLAS_INDEX n,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex * y, CBLAS_INDEX incy );

void cblas_zcopy(
    CBLAS_INDEX n,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex * y, CBLAS_INDEX incy );

float cblas_sdot(
    CBLAS_INDEX n,
    float const * x, CBLAS_INDEX incx,
    float const * y, CBLAS_INDEX incy );

double cblas_ddot(
    CBLAS_INDEX n,
    double const * x, CBLAS_INDEX incx,
    double const * y, CBLAS_INDEX incy );

float  _Complex cblas_cdot(
    CBLAS_INDEX n,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex const * y, CBLAS_INDEX incy );

double _Complex cblas_zdot(
    CBLAS_INDEX n,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex const * y, CBLAS_INDEX incy );

float cblas_sdotu(
    CBLAS_INDEX n,
    float const * x, CBLAS_INDEX incx,
    float const * y, CBLAS_INDEX incy );

double cblas_ddotu(
    CBLAS_INDEX n,
    double const * x, CBLAS_INDEX incx,
    double const * y, CBLAS_INDEX incy );

float  _Complex cblas_cdotu(
    CBLAS_INDEX n,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex const * y, CBLAS_INDEX incy );

double _Complex cblas_zdotu(
    CBLAS_INDEX n,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex const * y, CBLAS_INDEX incy );

CBLAS_INDEX cblas_siamax(
    CBLAS_INDEX n,
    float const * x, CBLAS_INDEX incx );

CBLAS_INDEX cblas_diamax(
    CBLAS_INDEX n,
    double const * x, CBLAS_INDEX incx );

CBLAS_INDEX cblas_ciamax(
    CBLAS_INDEX n,
    float  _Complex const * x, CBLAS_INDEX incx );

CBLAS_INDEX cblas_ziamax(
    CBLAS_INDEX n,
    double _Complex const * x, CBLAS_INDEX incx );

float cblas_snrm2(
    CBLAS_INDEX n,
    float const * x, CBLAS_INDEX incx );

double cblas_dnrm2(
    CBLAS_INDEX n,
    double const * x, CBLAS_INDEX incx );

float cblas_cnrm2(
    CBLAS_INDEX n,
    float  _Complex const * x, CBLAS_INDEX incx );

double cblas_znrm2(
    CBLAS_INDEX n,
    double _Complex const * x, CBLAS_INDEX incx );

void cblas_srot(
    CBLAS_INDEX n,
    float * x, CBLAS_INDEX incx,
    float * y, CBLAS_INDEX incy,
    float c,
    float s );

void cblas_drot(
    CBLAS_INDEX n,
    double * x, CBLAS_INDEX incx,
    double * y, CBLAS_INDEX incy,
    double c,
    double s );

void cblas_csrot(
    CBLAS_INDEX n,
    float  _Complex * x, CBLAS_INDEX incx,
    float  _Complex * y, CBLAS_INDEX incy,
    float c,
    float s );

void cblas_zdrot(
    CBLAS_INDEX n,
    double _Complex * x, CBLAS_INDEX incx,
    double _Complex * y, CBLAS_INDEX incy,
    double c,
    double s );

void cblas_crot(
    CBLAS_INDEX n,
    float  _Complex * x, CBLAS_INDEX incx,
    float  _Complex * y, CBLAS_INDEX incy,
    float c,
    float  _Complex s );

void cblas_zrot(
    CBLAS_INDEX n,
    double _Complex * x, CBLAS_INDEX incx,
    double _Complex * y, CBLAS_INDEX incy,
    double c,
    double _Complex s );

void cblas_srotg(
    float * a,
    float * b,
    float * c,
    float * s );

void cblas_drotg(
    double * a,
    double * b,
    double * c,
    double * s );

void cblas_crotg(
    float  _Complex * a,
    float  _Complex * b,
    float * c,
    float  _Complex * s );

void cblas_zrotg(
    double _Complex * a,
    double _Complex * b,
    double * c,
    double _Complex * s );

void cblas_srotm(
    CBLAS_INDEX n,
    float * x, CBLAS_INDEX incx,
    float * y, CBLAS_INDEX incy,
    float const * param );

void cblas_drotm(
    CBLAS_INDEX n,
    double * x, CBLAS_INDEX incx,
    double * y, CBLAS_INDEX incy,
    double const * param );

void cblas_srotmg(
    float * d1,
    float * d2,
    float * a,
    float b,
    float * param );

void cblas_drotmg(
    double * d1,
    double * d2,
    double * a,
    double b,
    double * param );

void cblas_sscal(
    CBLAS_INDEX n,
    float alpha,
    float * x, CBLAS_INDEX incx );

void cblas_dscal(
    CBLAS_INDEX n,
    double alpha,
    double * x, CBLAS_INDEX incx );

void cblas_cscal(
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex * x, CBLAS_INDEX incx );

void cblas_zscal(
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex * x, CBLAS_INDEX incx );

void cblas_sswap(
    CBLAS_INDEX n,
    float * x, CBLAS_INDEX incx,
    float * y, CBLAS_INDEX incy );

void cblas_dswap(
    CBLAS_INDEX n,
    double * x, CBLAS_INDEX incx,
    double * y, CBLAS_INDEX incy );

void cblas_cswap(
    CBLAS_INDEX n,
    float  _Complex * x, CBLAS_INDEX incx,
    float  _Complex * y, CBLAS_INDEX incy );

void cblas_zswap(
    CBLAS_INDEX n,
    double _Complex * x, CBLAS_INDEX incx,
    double _Complex * y, CBLAS_INDEX incy );

// =============================================================================
// Level 2 BLAS

void cblas_sgemv(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float const * x, CBLAS_INDEX incx,
    float beta,
    float * y, CBLAS_INDEX incy );

void cblas_dgemv(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double const * x, CBLAS_INDEX incx,
    double beta,
    double * y, CBLAS_INDEX incy );

void cblas_cgemv(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex beta,
    float  _Complex * y, CBLAS_INDEX incy );

void cblas_zgemv(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex beta,
    double _Complex * y, CBLAS_INDEX incy );

void cblas_sger(
    CBLAS_LAYOUT layout,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float alpha,
    float const * x, CBLAS_INDEX incx,
    float const * y, CBLAS_INDEX incy,
    float * A, CBLAS_INDEX lda );

void cblas_dger(
    CBLAS_LAYOUT layout,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double alpha,
    double const * x, CBLAS_INDEX incx,
    double const * y, CBLAS_INDEX incy,
    double * A, CBLAS_INDEX lda );

void cblas_cger(
    CBLAS_LAYOUT layout,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex const * y, CBLAS_INDEX incy,
    float  _Complex * A, CBLAS_INDEX lda );

void cblas_zger(
    CBLAS_LAYOUT layout,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex const * y, CBLAS_INDEX incy,
    double _Complex * A, CBLAS_INDEX lda );

void cblas_sgeru(
    CBLAS_LAYOUT layout,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float alpha,
    float const * x, CBLAS_INDEX incx,
    float const * y, CBLAS_INDEX incy,
    float * A, CBLAS_INDEX lda );

void cblas_dgeru(
    CBLAS_LAYOUT layout,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double alpha,
    double const * x, CBLAS_INDEX incx,
    double const * y, CBLAS_INDEX incy,
    double * A, CBLAS_INDEX lda );

void cblas_cgeru(
    CBLAS_LAYOUT layout,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex const * y, CBLAS_INDEX incy,
    float  _Complex * A, CBLAS_INDEX lda );

void cblas_zgeru(
    CBLAS_LAYOUT layout,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex const * y, CBLAS_INDEX incy,
    double _Complex * A, CBLAS_INDEX lda );

void cblas_shemv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float const * x, CBLAS_INDEX incx,
    float beta,
    float * y, CBLAS_INDEX incy );

void cblas_dhemv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double const * x, CBLAS_INDEX incx,
    double beta,
    double * y, CBLAS_INDEX incy );

void cblas_chemv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex beta,
    float  _Complex * y, CBLAS_INDEX incy );

void cblas_zhemv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex beta,
    double _Complex * y, CBLAS_INDEX incy );

void cblas_sher(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float alpha,
    float const * x, CBLAS_INDEX incx,
    float * A, CBLAS_INDEX lda );

void cblas_dher(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double alpha,
    double const * x, CBLAS_INDEX incx,
    double * A, CBLAS_INDEX lda );

void cblas_cher(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float alpha,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex * A, CBLAS_INDEX lda );

void cblas_zher(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double alpha,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex * A, CBLAS_INDEX lda );

void cblas_sher2(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float alpha,
    float const * x, CBLAS_INDEX incx,
    float const * y, CBLAS_INDEX incy,
    float * A, CBLAS_INDEX lda );

void cblas_dher2(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double alpha,
    double const * x, CBLAS_INDEX incx,
    double const * y, CBLAS_INDEX incy,
    double * A, CBLAS_INDEX lda );

void cblas_cher2(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex const * y, CBLAS_INDEX incy,
    float  _Complex * A, CBLAS_INDEX lda );

void cblas_zher2(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex const * y, CBLAS_INDEX incy,
    double _Complex * A, CBLAS_INDEX lda );

void cblas_ssymv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float const * x, CBLAS_INDEX incx,
    float beta,
    float * y, CBLAS_INDEX incy );

void cblas_dsymv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double const * x, CBLAS_INDEX incx,
    double beta,
    double * y, CBLAS_INDEX incy );

void cblas_csymv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex beta,
    float  _Complex * y, CBLAS_INDEX incy );

void cblas_zsymv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex beta,
    double _Complex * y, CBLAS_INDEX incy );

void cblas_ssyr(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float alpha,
    float const * x, CBLAS_INDEX incx,
    float * A, CBLAS_INDEX lda );

void cblas_dsyr(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double alpha,
    double const * x, CBLAS_INDEX incx,
    double * A, CBLAS_INDEX lda );

void cblas_ssyr2(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float alpha,
    float const * x, CBLAS_INDEX incx,
    float const * y, CBLAS_INDEX incy,
    float * A, CBLAS_INDEX lda );

void cblas_dsyr2(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double alpha,
    double const * x, CBLAS_INDEX incx,
    double const * y, CBLAS_INDEX incy,
    double * A, CBLAS_INDEX lda );

void cblas_csyr2(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * x, CBLAS_INDEX incx,
    float  _Complex const * y, CBLAS_INDEX incy,
    float  _Complex * A, CBLAS_INDEX lda );

void cblas_zsyr2(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * x, CBLAS_INDEX incx,
    double _Complex const * y, CBLAS_INDEX incy,
    double _Complex * A, CBLAS_INDEX lda );

void cblas_strmv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX n,
    float const * A, CBLAS_INDEX lda,
    float * x, CBLAS_INDEX incx );

void cblas_dtrmv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX n,
    double const * A, CBLAS_INDEX lda,
    double * x, CBLAS_INDEX incx );

void cblas_ctrmv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX n,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex * x, CBLAS_INDEX incx );

void cblas_ztrmv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX n,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex * x, CBLAS_INDEX incx );

void cblas_strsv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX n,
    float const * A, CBLAS_INDEX lda,
    float * x, CBLAS_INDEX incx );

void cblas_dtrsv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX n,
    double const * A, CBLAS_INDEX lda,
    double * x, CBLAS_INDEX incx );

void cblas_ctrsv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX n,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex * x, CBLAS_INDEX incx );

void cblas_ztrsv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX n,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex * x, CBLAS_INDEX incx );

// =============================================================================
// Level 3 BLAS

void cblas_sgemm(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float const * B, CBLAS_INDEX ldb,
    float beta,
    float * C, CBLAS_INDEX ldc );

void cblas_dgemm(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double const * B, CBLAS_INDEX ldb,
    double beta,
    double * C, CBLAS_INDEX ldc );

void cblas_cgemm(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex const * B, CBLAS_INDEX ldb,
    float  _Complex beta,
    float  _Complex * C, CBLAS_INDEX ldc );

void cblas_zgemm(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex const * B, CBLAS_INDEX ldb,
    double _Complex beta,
    double _Complex * C, CBLAS_INDEX ldc );

void cblas_shemm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float const * B, CBLAS_INDEX ldb,
    float beta,
    float * C, CBLAS_INDEX ldc );

void cblas_dhemm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double const * B, CBLAS_INDEX ldb,
    double beta,
    double * C, CBLAS_INDEX ldc );

void cblas_chemm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex const * B, CBLAS_INDEX ldb,
    float  _Complex beta,
    float  _Complex * C, CBLAS_INDEX ldc );

void cblas_zhemm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex const * B, CBLAS_INDEX ldb,
    double _Complex beta,
    double _Complex * C, CBLAS_INDEX ldc );

void cblas_sher2k(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float const * B, CBLAS_INDEX ldb,
    float beta,
    float * C, CBLAS_INDEX ldc );

void cblas_dher2k(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double const * B, CBLAS_INDEX ldb,
    double beta,
    double * C, CBLAS_INDEX ldc );

void cblas_cher2k(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex const * B, CBLAS_INDEX ldb,
    float beta,
    float  _Complex * C, CBLAS_INDEX ldc );

void cblas_zher2k(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex const * B, CBLAS_INDEX ldb,
    double beta,
    double _Complex * C, CBLAS_INDEX ldc );

void cblas_sherk(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float beta,
    float * C, CBLAS_INDEX ldc );

void cblas_dherk(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double beta,
    double * C, CBLAS_INDEX ldc );

void cblas_cherk(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    float alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float beta,
    float  _Complex * C, CBLAS_INDEX ldc );

void cblas_zherk(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    double alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double beta,
    double _Complex * C, CBLAS_INDEX ldc );

void cblas_ssymm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float const * B, CBLAS_INDEX ldb,
    float beta,
    float * C, CBLAS_INDEX ldc );

void cblas_dsymm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double const * B, CBLAS_INDEX ldb,
    double beta,
    double * C, CBLAS_INDEX ldc );

void cblas_csymm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex const * B, CBLAS_INDEX ldb,
    float  _Complex beta,
    float  _Complex * C, CBLAS_INDEX ldc );

void cblas_zsymm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex const * B, CBLAS_INDEX ldb,
    double _Complex beta,
    double _Complex * C, CBLAS_INDEX ldc );

void cblas_ssyr2k(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float const * B, CBLAS_INDEX ldb,
    float beta,
    float * C, CBLAS_INDEX ldc );

void cblas_dsyr2k(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double const * B, CBLAS_INDEX ldb,
    double beta,
    double * C, CBLAS_INDEX ldc );

void cblas_csyr2k(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex const * B, CBLAS_INDEX ldb,
    float  _Complex beta,
    float  _Complex * C, CBLAS_INDEX ldc );

void cblas_zsyr2k(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex const * B, CBLAS_INDEX ldb,
    double _Complex beta,
    double _Complex * C, CBLAS_INDEX ldc );

void cblas_ssyrk(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float beta,
    float * C, CBLAS_INDEX ldc );

void cblas_dsyrk(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double beta,
    double * C, CBLAS_INDEX ldc );

void cblas_csyrk(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex beta,
    float  _Complex * C, CBLAS_INDEX ldc );

void cblas_zsyrk(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_INDEX n,
    CBLAS_INDEX k,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex beta,
    double _Complex * C, CBLAS_INDEX ldc );

void cblas_strmm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float * B, CBLAS_INDEX ldb );

void cblas_dtrmm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double * B, CBLAS_INDEX ldb );

void cblas_ctrmm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex * B, CBLAS_INDEX ldb );

void cblas_ztrmm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex * B, CBLAS_INDEX ldb );

void cblas_strsm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float alpha,
    float const * A, CBLAS_INDEX lda,
    float * B, CBLAS_INDEX ldb );

void cblas_dtrsm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double alpha,
    double const * A, CBLAS_INDEX lda,
    double * B, CBLAS_INDEX ldb );

void cblas_ctrsm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    float  _Complex alpha,
    float  _Complex const * A, CBLAS_INDEX lda,
    float  _Complex * B, CBLAS_INDEX ldb );

void cblas_ztrsm(
    CBLAS_LAYOUT layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    CBLAS_INDEX m,
    CBLAS_INDEX n,
    double _Complex alpha,
    double _Complex const * A, CBLAS_INDEX lda,
    double _Complex * B, CBLAS_INDEX ldb );

#ifdef __cplusplus
}
#endif

#endif // __CBLAS_H__