#include "tblas.hpp"
#include "blas.h"

// -----------------------------------------------------------------------------
typedef std::complex<float> tblas_complexFloat;
typedef std::complex<double> tblas_complexDouble;

#define tblas_cteC(z) reinterpret_cast<const tblas_complexFloat*>( z )
#define tblas_cteZ(z) reinterpret_cast<const tblas_complexDouble*>( z )
#define tblas_C(z) reinterpret_cast<tblas_complexFloat*>( z )
#define tblas_Z(z) reinterpret_cast<tblas_complexDouble*>(  z )
// -----------------------------------------------------------------------------

extern "C" {

float sasum(
    int64_t n,
    float const * x, int64_t incx ) {
    return blas::asum<float>(
        n,
        x, incx );
}

double dasum(
    int64_t n,
    double const * x, int64_t incx ) {
    return blas::asum<double>(
        n,
        x, incx );
}

float casum(
    int64_t n,
    complexFloat const * x, int64_t incx ) {
    return blas::asum<tblas_complexFloat>(
        n,
        tblas_cteC(x), incx );
}

double zasum(
    int64_t n,
    complexDouble const * x, int64_t incx ) {
    return blas::asum<tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx );
}

void saxpy(
    int64_t n,
    float alpha,
    float const * x, int64_t incx,
    float * y, int64_t incy ) {
    return blas::axpy<float, float>(
        n,
        alpha,
        x, incx,
        y, incy );
}

void daxpy(
    int64_t n,
    double alpha,
    double const * x, int64_t incx,
    double * y, int64_t incy ) {
    return blas::axpy<double, double>(
        n,
        alpha,
        x, incx,
        y, incy );
}

void caxpy(
    int64_t n,
    complexFloat alpha,
    complexFloat const * x, int64_t incx,
    complexFloat * y, int64_t incy ) {
    return blas::axpy<tblas_complexFloat, tblas_complexFloat>(
        n,
        *tblas_C(&alpha),
        tblas_cteC(x), incx,
        tblas_C(y), incy );
}

void zaxpy(
    int64_t n,
    complexDouble alpha,
    complexDouble const * x, int64_t incx,
    complexDouble * y, int64_t incy ) {
    return blas::axpy<tblas_complexDouble, tblas_complexDouble>(
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(x), incx,
        tblas_Z(y), incy );
}

void scopy(
    int64_t n,
    float const * x, int64_t incx,
    float * y, int64_t incy ) {
    return blas::copy<float, float>(
        n,
        x, incx,
        y, incy );
}

void dcopy(
    int64_t n,
    double const * x, int64_t incx,
    double * y, int64_t incy ) {
    return blas::copy<double, double>(
        n,
        x, incx,
        y, incy );
}

void ccopy(
    int64_t n,
    complexFloat const * x, int64_t incx,
    complexFloat * y, int64_t incy ) {
    return blas::copy<tblas_complexFloat, tblas_complexFloat>(
        n,
        tblas_cteC(x), incx,
        tblas_C(y), incy );
}

void zcopy(
    int64_t n,
    complexDouble const * x, int64_t incx,
    complexDouble * y, int64_t incy ) {
    return blas::copy<tblas_complexDouble, tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx,
        tblas_Z(y), incy );
}

float sdot(
    int64_t n,
    float const * x, int64_t incx,
    float const * y, int64_t incy ) {
    return blas::dot<float, float>(
        n,
        x, incx,
        y, incy );
}

double ddot(
    int64_t n,
    double const * x, int64_t incx,
    double const * y, int64_t incy ) {
    return blas::dot<double, double>(
        n,
        x, incx,
        y, incy );
}

complexFloat cdot(
    int64_t n,
    complexFloat const * x, int64_t incx,
    complexFloat const * y, int64_t incy ) {
    tblas_complexFloat c = blas::dot<tblas_complexFloat, tblas_complexFloat>(
        n,
        tblas_cteC(x), incx,
        tblas_cteC(y), incy );
    return *reinterpret_cast<complexFloat*>(&c);
}

complexDouble zdot(
    int64_t n,
    complexDouble const * x, int64_t incx,
    complexDouble const * y, int64_t incy ) {
    tblas_complexDouble z = blas::dot<tblas_complexDouble, tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy );
    return *reinterpret_cast<complexDouble*>(&z);
}

float sdotu(
    int64_t n,
    float const * x, int64_t incx,
    float const * y, int64_t incy ) {
    return blas::dotu<float, float>(
        n,
        x, incx,
        y, incy );
}

double ddotu(
    int64_t n,
    double const * x, int64_t incx,
    double const * y, int64_t incy ) {
    return blas::dotu<double, double>(
        n,
        x, incx,
        y, incy );
}

complexFloat cdotu(
    int64_t n,
    complexFloat const * x, int64_t incx,
    complexFloat const * y, int64_t incy ) {
    tblas_complexDouble c = blas::dotu<tblas_complexFloat, tblas_complexFloat>(
        n,
        tblas_cteC(x), incx,
        tblas_cteC(y), incy );
    return *reinterpret_cast<complexFloat*>(&c);
}

complexDouble zdotu(
    int64_t n,
    complexDouble const * x, int64_t incx,
    complexDouble const * y, int64_t incy ) {
    tblas_complexDouble z = blas::dotu<tblas_complexDouble, tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy );
    return *reinterpret_cast<complexDouble*>(&z);
}

int64_t siamax(
    int64_t n,
    float const * x, int64_t incx ) {
    return blas::iamax<float>(
        n,
        x, incx );
}

int64_t diamax(
    int64_t n,
    double const * x, int64_t incx ) {
    return blas::iamax<double>(
        n,
        x, incx );
}

int64_t ciamax(
    int64_t n,
    complexFloat const * x, int64_t incx ) {
    return blas::iamax<tblas_complexFloat>(
        n,
        tblas_cteC(x), incx );
}

int64_t ziamax(
    int64_t n,
    complexDouble const * x, int64_t incx ) {
    return blas::iamax<tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx );
}

float snrm2(
    int64_t n,
    float const * x, int64_t incx ) {
    return blas::nrm2<float>(
        n,
        x, incx );
}

double dnrm2(
    int64_t n,
    double const * x, int64_t incx ) {
    return blas::nrm2<double>(
        n,
        x, incx );
}

float cnrm2(
    int64_t n,
    complexFloat const * x, int64_t incx ) {
    return blas::nrm2<tblas_complexFloat>(
        n,
        tblas_cteC(x), incx );
}

double znrm2(
    int64_t n,
    complexDouble const * x, int64_t incx ) {
    return blas::nrm2<tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx );
}

void srot(
    int64_t n,
    float * x, int64_t incx,
    float * y, int64_t incy,
    float c,
    float s ) {
    return blas::rot<float, float>(
        n,
        x, incx,
        y, incy,
        c,
        s );
}

void drot(
    int64_t n,
    double * x, int64_t incx,
    double * y, int64_t incy,
    double c,
    double s ) {
    return blas::rot<double, double>(
        n,
        x, incx,
        y, incy,
        c,
        s );
}

void csrot(
    int64_t n,
    complexFloat * x, int64_t incx,
    complexFloat * y, int64_t incy,
    float c,
    float s ) {
    return blas::rot(
        n,
        tblas_C(x), incx,
        tblas_C(y), incy,
        c,
        s );
}

void zdrot(
    int64_t n,
    complexDouble * x, int64_t incx,
    complexDouble * y, int64_t incy,
    double c,
    double s ) {
    return blas::rot(
        n,
        tblas_Z(x), incx,
        tblas_Z(y), incy,
        c,
        s );
}

void crot(
    int64_t n,
    complexFloat * x, int64_t incx,
    complexFloat * y, int64_t incy,
    float c,
    complexFloat s ) {
    return blas::rot<tblas_complexFloat, tblas_complexFloat>(
        n,
        tblas_C(x), incx,
        tblas_C(y), incy,
        c,
        *tblas_C(&s) );
}

void zrot(
    int64_t n,
    complexDouble * x, int64_t incx,
    complexDouble * y, int64_t incy,
    double c,
    complexDouble s ) {
    return blas::rot<tblas_complexDouble, tblas_complexDouble>(
        n,
        tblas_Z(x), incx,
        tblas_Z(y), incy,
        c,
        *tblas_Z(&s) );
}

void srotg(
    float * a,
    float * b,
    float * c,
    float * s ) {
    return blas::rotg<float, float>(
        a,
        b,
        c,
        s );
}

void drotg(
    double * a,
    double * b,
    double * c,
    double * s ) {
    return blas::rotg<double, double>(
        a,
        b,
        c,
        s );
}

void crotg(
    complexFloat * a,
    complexFloat * b,
    float * c,
    complexFloat * s ) {
    return blas::rotg<tblas_complexFloat, tblas_complexFloat>(
        tblas_C(a),
        tblas_C(b),
        c,
        tblas_C(s) );
}

void zrotg(
    complexDouble * a,
    complexDouble * b,
    double * c,
    complexDouble * s ) {
    return blas::rotg<tblas_complexDouble, tblas_complexDouble>(
        tblas_Z(a),
        tblas_Z(b),
        c,
        tblas_Z(s) );
}

void srotm(
    int64_t n,
    float * x, int64_t incx,
    float * y, int64_t incy,
    float const * param ) {
    return blas::rotm<float, float>(
        n,
        x, incx,
        y, incy,
        param );
}

void drotm(
    int64_t n,
    double * x, int64_t incx,
    double * y, int64_t incy,
    double const * param ) {
    return blas::rotm<double, double>(
        n,
        x, incx,
        y, incy,
        param );
}

void srotmg(
    float * d1,
    float * d2,
    float * a,
    float b,
    float * param ) {
    return blas::rotmg<float>(
        d1,
        d2,
        a,
        b,
        param );
}

void drotmg(
    double * d1,
    double * d2,
    double * a,
    double b,
    double * param ) {
    return blas::rotmg<double>(
        d1,
        d2,
        a,
        b,
        param );
}

void sscal(
    int64_t n,
    float alpha,
    float * x, int64_t incx ) {
    return blas::scal<float>(
        n,
        alpha,
        x, incx );
}

void dscal(
    int64_t n,
    double alpha,
    double * x, int64_t incx ) {
    return blas::scal<double>(
        n,
        alpha,
        x, incx );
}

void cscal(
    int64_t n,
    complexFloat alpha,
    complexFloat * x, int64_t incx ) {
    return blas::scal<tblas_complexFloat>(
        n,
        *tblas_C(&alpha),
        tblas_C(x), incx );
}

void zscal(
    int64_t n,
    complexDouble alpha,
    complexDouble * x, int64_t incx ) {
    return blas::scal<tblas_complexDouble>(
        n,
        *tblas_Z(&alpha),
        tblas_Z(x), incx );
}

void sswap(
    int64_t n,
    float * x, int64_t incx,
    float * y, int64_t incy ) {
    return blas::swap<float, float>(
        n,
        x, incx,
        y, incy );
}

void dswap(
    int64_t n,
    double * x, int64_t incx,
    double * y, int64_t incy ) {
    return blas::swap<double, double>(
        n,
        x, incx,
        y, incy );
}

void cswap(
    int64_t n,
    complexFloat * x, int64_t incx,
    complexFloat * y, int64_t incy ) {
    return blas::swap<tblas_complexFloat, tblas_complexFloat>(
        n,
        tblas_C(x), incx,
        tblas_C(y), incy );
}

void zswap(
    int64_t n,
    complexDouble * x, int64_t incx,
    complexDouble * y, int64_t incy ) {
    return blas::swap<tblas_complexDouble, tblas_complexDouble>(
        n,
        tblas_Z(x), incx,
        tblas_Z(y), incy );
}

void sgemv(
    Layout layout,
    Op trans,
    int64_t m,
    int64_t n,
    float alpha,
    float const * A, int64_t lda,
    float const * x, int64_t incx,
    float beta,
    float * y, int64_t incy ) {
    return blas::gemv<float, float, float>(
        (blas::Layout) layout,
        (blas::Op) trans,
        m,
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

void dgemv(
    Layout layout,
    Op trans,
    int64_t m,
    int64_t n,
    double alpha,
    double const * A, int64_t lda,
    double const * x, int64_t incx,
    double beta,
    double * y, int64_t incy ) {
    return blas::gemv<double, double, double>(
        (blas::Layout) layout,
        (blas::Op) trans,
        m,
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

void cgemv(
    Layout layout,
    Op trans,
    int64_t m,
    int64_t n,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat const * x, int64_t incx,
    complexFloat beta,
    complexFloat * y, int64_t incy ) {
    return blas::gemv<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Op) trans,
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(x), incx,
        *tblas_C(&beta),
        tblas_C(y), incy );
}

void zgemv(
    Layout layout,
    Op trans,
    int64_t m,
    int64_t n,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble const * x, int64_t incx,
    complexDouble beta,
    complexDouble * y, int64_t incy ) {
    return blas::gemv<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Op) trans,
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(x), incx,
        *tblas_Z(&beta),
        tblas_Z(y), incy );
}

void sger(
    Layout layout,
    int64_t m,
    int64_t n,
    float alpha,
    float const * x, int64_t incx,
    float const * y, int64_t incy,
    float * A, int64_t lda ) {
    return blas::ger<float, float, float>(
        (blas::Layout) layout,
        m,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

void dger(
    Layout layout,
    int64_t m,
    int64_t n,
    double alpha,
    double const * x, int64_t incx,
    double const * y, int64_t incy,
    double * A, int64_t lda ) {
    return blas::ger<double, double, double>(
        (blas::Layout) layout,
        m,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

void cger(
    Layout layout,
    int64_t m,
    int64_t n,
    complexFloat alpha,
    complexFloat const * x, int64_t incx,
    complexFloat const * y, int64_t incy,
    complexFloat * A, int64_t lda ) {
    return blas::ger<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(x), incx,
        tblas_cteC(y), incy,
        tblas_C(A), lda );
}

void zger(
    Layout layout,
    int64_t m,
    int64_t n,
    complexDouble alpha,
    complexDouble const * x, int64_t incx,
    complexDouble const * y, int64_t incy,
    complexDouble * A, int64_t lda ) {
    return blas::ger<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy,
        tblas_Z(A), lda );
}

void sgeru(
    Layout layout,
    int64_t m,
    int64_t n,
    float alpha,
    float const * x, int64_t incx,
    float const * y, int64_t incy,
    float * A, int64_t lda ) {
    return blas::geru<float, float, float>(
        (blas::Layout) layout,
        m,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

void dgeru(
    Layout layout,
    int64_t m,
    int64_t n,
    double alpha,
    double const * x, int64_t incx,
    double const * y, int64_t incy,
    double * A, int64_t lda ) {
    return blas::geru<double, double, double>(
        (blas::Layout) layout,
        m,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

void cgeru(
    Layout layout,
    int64_t m,
    int64_t n,
    complexFloat alpha,
    complexFloat const * x, int64_t incx,
    complexFloat const * y, int64_t incy,
    complexFloat * A, int64_t lda ) {
    return blas::geru<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(x), incx,
        tblas_cteC(y), incy,
        tblas_C(A), lda );
}

void zgeru(
    Layout layout,
    int64_t m,
    int64_t n,
    complexDouble alpha,
    complexDouble const * x, int64_t incx,
    complexDouble const * y, int64_t incy,
    complexDouble * A, int64_t lda ) {
    return blas::geru<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy,
        tblas_Z(A), lda );
}

void shemv(
    Layout layout,
    Uplo uplo,
    int64_t n,
    float alpha,
    float const * A, int64_t lda,
    float const * x, int64_t incx,
    float beta,
    float * y, int64_t incy ) {
    return blas::hemv<float, float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

void dhemv(
    Layout layout,
    Uplo uplo,
    int64_t n,
    double alpha,
    double const * A, int64_t lda,
    double const * x, int64_t incx,
    double beta,
    double * y, int64_t incy ) {
    return blas::hemv<double, double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

void chemv(
    Layout layout,
    Uplo uplo,
    int64_t n,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat const * x, int64_t incx,
    complexFloat beta,
    complexFloat * y, int64_t incy ) {
    return blas::hemv<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(x), incx,
        *tblas_C(&beta),
        tblas_C(y), incy );
}

void zhemv(
    Layout layout,
    Uplo uplo,
    int64_t n,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble const * x, int64_t incx,
    complexDouble beta,
    complexDouble * y, int64_t incy ) {
    return blas::hemv<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(x), incx,
        *tblas_Z(&beta),
        tblas_Z(y), incy );
}

void sher(
    Layout layout,
    Uplo uplo,
    int64_t n,
    float alpha,
    float const * x, int64_t incx,
    float * A, int64_t lda ) {
    return blas::her<float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        x, incx,
        A, lda );
}

void dher(
    Layout layout,
    Uplo uplo,
    int64_t n,
    double alpha,
    double const * x, int64_t incx,
    double * A, int64_t lda ) {
    return blas::her<double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        x, incx,
        A, lda );
}

void cher(
    Layout layout,
    Uplo uplo,
    int64_t n,
    float alpha,
    complexFloat const * x, int64_t incx,
    complexFloat * A, int64_t lda ) {
    return blas::her<tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        tblas_cteC(x), incx,
        tblas_C(A), lda );
}

void zher(
    Layout layout,
    Uplo uplo,
    int64_t n,
    double alpha,
    complexDouble const * x, int64_t incx,
    complexDouble * A, int64_t lda ) {
    return blas::her<tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        tblas_cteZ(x), incx,
        tblas_Z(A), lda );
}

void sher2(
    Layout layout,
    Uplo uplo,
    int64_t n,
    float alpha,
    float const * x, int64_t incx,
    float const * y, int64_t incy,
    float * A, int64_t lda ) {
    return blas::her2<float, float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

void dher2(
    Layout layout,
    Uplo uplo,
    int64_t n,
    double alpha,
    double const * x, int64_t incx,
    double const * y, int64_t incy,
    double * A, int64_t lda ) {
    return blas::her2<double, double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

void cher2(
    Layout layout,
    Uplo uplo,
    int64_t n,
    complexFloat alpha,
    complexFloat const * x, int64_t incx,
    complexFloat const * y, int64_t incy,
    complexFloat * A, int64_t lda ) {
    return blas::her2<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        *tblas_C(&alpha),
        tblas_cteC(x), incx,
        tblas_cteC(y), incy,
        tblas_C(A), lda );
}

void zher2(
    Layout layout,
    Uplo uplo,
    int64_t n,
    complexDouble alpha,
    complexDouble const * x, int64_t incx,
    complexDouble const * y, int64_t incy,
    complexDouble * A, int64_t lda ) {
    return blas::her2<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy,
        tblas_Z(A), lda );
}

void ssymv(
    Layout layout,
    Uplo uplo,
    int64_t n,
    float alpha,
    float const * A, int64_t lda,
    float const * x, int64_t incx,
    float beta,
    float * y, int64_t incy ) {
    return blas::symv<float, float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

void dsymv(
    Layout layout,
    Uplo uplo,
    int64_t n,
    double alpha,
    double const * A, int64_t lda,
    double const * x, int64_t incx,
    double beta,
    double * y, int64_t incy ) {
    return blas::symv<double, double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

void csymv(
    Layout layout,
    Uplo uplo,
    int64_t n,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat const * x, int64_t incx,
    complexFloat beta,
    complexFloat * y, int64_t incy ) {
    return blas::symv<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(x), incx,
        *tblas_C(&beta),
        tblas_C(y), incy );
}

void zsymv(
    Layout layout,
    Uplo uplo,
    int64_t n,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble const * x, int64_t incx,
    complexDouble beta,
    complexDouble * y, int64_t incy ) {
    return blas::symv<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(x), incx,
        *tblas_Z(&beta),
        tblas_Z(y), incy );
}

void ssyr(
    Layout layout,
    Uplo uplo,
    int64_t n,
    float alpha,
    float const * x, int64_t incx,
    float * A, int64_t lda ) {
    return blas::syr<float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        x, incx,
        A, lda );
}

void dsyr(
    Layout layout,
    Uplo uplo,
    int64_t n,
    double alpha,
    double const * x, int64_t incx,
    double * A, int64_t lda ) {
    return blas::syr<double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        x, incx,
        A, lda );
}

void ssyr2(
    Layout layout,
    Uplo uplo,
    int64_t n,
    float alpha,
    float const * x, int64_t incx,
    float const * y, int64_t incy,
    float * A, int64_t lda ) {
    return blas::syr2<float, float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

void dsyr2(
    Layout layout,
    Uplo uplo,
    int64_t n,
    double alpha,
    double const * x, int64_t incx,
    double const * y, int64_t incy,
    double * A, int64_t lda ) {
    return blas::syr2<double, double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

void csyr2(
    Layout layout,
    Uplo uplo,
    int64_t n,
    complexFloat alpha,
    complexFloat const * x, int64_t incx,
    complexFloat const * y, int64_t incy,
    complexFloat * A, int64_t lda ) {
    return blas::syr2<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        *tblas_C(&alpha),
        tblas_cteC(x), incx,
        tblas_cteC(y), incy,
        tblas_C(A), lda );
}

void zsyr2(
    Layout layout,
    Uplo uplo,
    int64_t n,
    complexDouble alpha,
    complexDouble const * x, int64_t incx,
    complexDouble const * y, int64_t incy,
    complexDouble * A, int64_t lda ) {
    return blas::syr2<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy,
        tblas_Z(A), lda );
}

void strmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t n,
    float const * A, int64_t lda,
    float * x, int64_t incx ) {
    return blas::trmv<float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        n,
        A, lda,
        x, incx );
}

void dtrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t n,
    double const * A, int64_t lda,
    double * x, int64_t incx ) {
    return blas::trmv<double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        n,
        A, lda,
        x, incx );
}

void ctrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t n,
    complexFloat const * A, int64_t lda,
    complexFloat * x, int64_t incx ) {
    return blas::trmv<tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        n,
        tblas_cteC(A), lda,
        tblas_C(x), incx );
}

void ztrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t n,
    complexDouble const * A, int64_t lda,
    complexDouble * x, int64_t incx ) {
    return blas::trmv<tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        n,
        tblas_cteZ(A), lda,
        tblas_Z(x), incx );
}

void strsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t n,
    float const * A, int64_t lda,
    float * x, int64_t incx ) {
    return blas::trsv<float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        n,
        A, lda,
        x, incx );
}

void dtrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t n,
    double const * A, int64_t lda,
    double * x, int64_t incx ) {
    return blas::trsv<double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        n,
        A, lda,
        x, incx );
}

void ctrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t n,
    complexFloat const * A, int64_t lda,
    complexFloat * x, int64_t incx ) {
    return blas::trsv<tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        n,
        tblas_cteC(A), lda,
        tblas_C(x), incx );
}

void ztrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t n,
    complexDouble const * A, int64_t lda,
    complexDouble * x, int64_t incx ) {
    return blas::trsv<tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        n,
        tblas_cteZ(A), lda,
        tblas_Z(x), incx );
}

void sgemm(
    Layout layout,
    Op transA,
    Op transB,
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    float const * A, int64_t lda,
    float const * B, int64_t ldb,
    float beta,
    float * C, int64_t ldc ) {
    return blas::gemm<float, float, float>(
        (blas::Layout) layout,
        (blas::Op) transA,
        (blas::Op) transB,
        m,
        n,
        k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

void dgemm(
    Layout layout,
    Op transA,
    Op transB,
    int64_t m,
    int64_t n,
    int64_t k,
    double alpha,
    double const * A, int64_t lda,
    double const * B, int64_t ldb,
    double beta,
    double * C, int64_t ldc ) {
    return blas::gemm<double, double, double>(
        (blas::Layout) layout,
        (blas::Op) transA,
        (blas::Op) transB,
        m,
        n,
        k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

void cgemm(
    Layout layout,
    Op transA,
    Op transB,
    int64_t m,
    int64_t n,
    int64_t k,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat const * B, int64_t ldb,
    complexFloat beta,
    complexFloat * C, int64_t ldc ) {
    return blas::gemm<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Op) transA,
        (blas::Op) transB,
        m,
        n,
        k,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(B), ldb,
        *tblas_C(&beta),
        tblas_C(C), ldc );
}

void zgemm(
    Layout layout,
    Op transA,
    Op transB,
    int64_t m,
    int64_t n,
    int64_t k,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble const * B, int64_t ldb,
    complexDouble beta,
    complexDouble * C, int64_t ldc ) {
    return blas::gemm<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Op) transA,
        (blas::Op) transB,
        m,
        n,
        k,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(B), ldb,
        *tblas_Z(&beta),
        tblas_Z(C), ldc );
}

void shemm(
    Layout layout,
    Side side,
    Uplo uplo,
    int64_t m,
    int64_t n,
    float alpha,
    float const * A, int64_t lda,
    float const * B, int64_t ldb,
    float beta,
    float * C, int64_t ldc ) {
    return blas::hemm<float, float, float>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        m,
        n,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

void dhemm(
    Layout layout,
    Side side,
    Uplo uplo,
    int64_t m,
    int64_t n,
    double alpha,
    double const * A, int64_t lda,
    double const * B, int64_t ldb,
    double beta,
    double * C, int64_t ldc ) {
    return blas::hemm<double, double, double>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        m,
        n,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

void chemm(
    Layout layout,
    Side side,
    Uplo uplo,
    int64_t m,
    int64_t n,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat const * B, int64_t ldb,
    complexFloat beta,
    complexFloat * C, int64_t ldc ) {
    return blas::hemm<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(B), ldb,
        *tblas_C(&beta),
        tblas_C(C), ldc );
}

void zhemm(
    Layout layout,
    Side side,
    Uplo uplo,
    int64_t m,
    int64_t n,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble const * B, int64_t ldb,
    complexDouble beta,
    complexDouble * C, int64_t ldc ) {
    return blas::hemm<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(B), ldb,
        *tblas_Z(&beta),
        tblas_Z(C), ldc );
}

void sher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    float alpha,
    float const * A, int64_t lda,
    float const * B, int64_t ldb,
    float beta,
    float * C, int64_t ldc ) {
    return blas::her2k<float, float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

void dher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    double alpha,
    double const * A, int64_t lda,
    double const * B, int64_t ldb,
    double beta,
    double * C, int64_t ldc ) {
    return blas::her2k<double, double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

void cher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat const * B, int64_t ldb,
    float beta,
    complexFloat * C, int64_t ldc ) {
    return blas::her2k<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(B), ldb,
        beta,
        tblas_C(C), ldc );
}

void zher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble const * B, int64_t ldb,
    double beta,
    complexDouble * C, int64_t ldc ) {
    return blas::her2k<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(B), ldb,
        beta,
        tblas_Z(C), ldc );
}

void sherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    float alpha,
    float const * A, int64_t lda,
    float beta,
    float * C, int64_t ldc ) {
    return blas::herk<float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        alpha,
        A, lda,
        beta,
        C, ldc );
}

void dherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    double alpha,
    double const * A, int64_t lda,
    double beta,
    double * C, int64_t ldc ) {
    return blas::herk<double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        alpha,
        A, lda,
        beta,
        C, ldc );
}

void cherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    float alpha,
    complexFloat const * A, int64_t lda,
    float beta,
    complexFloat * C, int64_t ldc ) {
    return blas::herk<tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        alpha,
        tblas_cteC(A), lda,
        beta,
        tblas_C(C), ldc );
}

void zherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    double alpha,
    complexDouble const * A, int64_t lda,
    double beta,
    complexDouble * C, int64_t ldc ) {
    return blas::herk<tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        alpha,
        tblas_cteZ(A), lda,
        beta,
        tblas_Z(C), ldc );
}

void ssymm(
    Layout layout,
    Side side,
    Uplo uplo,
    int64_t m,
    int64_t n,
    float alpha,
    float const * A, int64_t lda,
    float const * B, int64_t ldb,
    float beta,
    float * C, int64_t ldc ) {
    return blas::symm<float, float, float>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        m,
        n,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

void dsymm(
    Layout layout,
    Side side,
    Uplo uplo,
    int64_t m,
    int64_t n,
    double alpha,
    double const * A, int64_t lda,
    double const * B, int64_t ldb,
    double beta,
    double * C, int64_t ldc ) {
    return blas::symm<double, double, double>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        m,
        n,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

void csymm(
    Layout layout,
    Side side,
    Uplo uplo,
    int64_t m,
    int64_t n,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat const * B, int64_t ldb,
    complexFloat beta,
    complexFloat * C, int64_t ldc ) {
    return blas::symm<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(B), ldb,
        *tblas_C(&beta),
        tblas_C(C), ldc );
}

void zsymm(
    Layout layout,
    Side side,
    Uplo uplo,
    int64_t m,
    int64_t n,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble const * B, int64_t ldb,
    complexDouble beta,
    complexDouble * C, int64_t ldc ) {
    return blas::symm<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(B), ldb,
        *tblas_Z(&beta),
        tblas_Z(C), ldc );
}

void ssyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    float alpha,
    float const * A, int64_t lda,
    float const * B, int64_t ldb,
    float beta,
    float * C, int64_t ldc ) {
    return blas::syr2k<float, float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

void dsyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    double alpha,
    double const * A, int64_t lda,
    double const * B, int64_t ldb,
    double beta,
    double * C, int64_t ldc ) {
    return blas::syr2k<double, double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

void csyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat const * B, int64_t ldb,
    complexFloat beta,
    complexFloat * C, int64_t ldc ) {
    return blas::syr2k<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(B), ldb,
        *tblas_C(&beta),
        tblas_C(C), ldc );
}

void zsyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble const * B, int64_t ldb,
    complexDouble beta,
    complexDouble * C, int64_t ldc ) {
    return blas::syr2k<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(B), ldb,
        *tblas_Z(&beta),
        tblas_Z(C), ldc );
}

void ssyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    float alpha,
    float const * A, int64_t lda,
    float beta,
    float * C, int64_t ldc ) {
    return blas::syrk<float, float>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        alpha,
        A, lda,
        beta,
        C, ldc );
}

void dsyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    double alpha,
    double const * A, int64_t lda,
    double beta,
    double * C, int64_t ldc ) {
    return blas::syrk<double, double>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        alpha,
        A, lda,
        beta,
        C, ldc );
}

void csyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat beta,
    complexFloat * C, int64_t ldc ) {
    return blas::syrk<tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        *tblas_C(&beta),
        tblas_C(C), ldc );
}

void zsyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n,
    int64_t k,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble beta,
    complexDouble * C, int64_t ldc ) {
    return blas::syrk<tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        n,
        k,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        *tblas_Z(&beta),
        tblas_Z(C), ldc );
}

void strmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t m,
    int64_t n,
    float alpha,
    float const * A, int64_t lda,
    float * B, int64_t ldb ) {
    return blas::trmm<float, float>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        m,
        n,
        alpha,
        A, lda,
        B, ldb );
}

void dtrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t m,
    int64_t n,
    double alpha,
    double const * A, int64_t lda,
    double * B, int64_t ldb ) {
    return blas::trmm<double, double>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        m,
        n,
        alpha,
        A, lda,
        B, ldb );
}

void ctrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t m,
    int64_t n,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat * B, int64_t ldb ) {
    return blas::trmm<tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_C(B), ldb );
}

void ztrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t m,
    int64_t n,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble * B, int64_t ldb ) {
    return blas::trmm<tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_Z(B), ldb );
}

void strsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t m,
    int64_t n,
    float alpha,
    float const * A, int64_t lda,
    float * B, int64_t ldb ) {
    return blas::trsm<float, float>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        m,
        n,
        alpha,
        A, lda,
        B, ldb );
}

void dtrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t m,
    int64_t n,
    double alpha,
    double const * A, int64_t lda,
    double * B, int64_t ldb ) {
    return blas::trsm<double, double>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        m,
        n,
        alpha,
        A, lda,
        B, ldb );
}

void ctrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t m,
    int64_t n,
    complexFloat alpha,
    complexFloat const * A, int64_t lda,
    complexFloat * B, int64_t ldb ) {
    return blas::trsm<tblas_complexFloat, tblas_complexFloat>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_C(B), ldb );
}

void ztrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t m,
    int64_t n,
    complexDouble alpha,
    complexDouble const * A, int64_t lda,
    complexDouble * B, int64_t ldb ) {
    return blas::trsm<tblas_complexDouble, tblas_complexDouble>(
        (blas::Layout) layout,
        (blas::Side) side,
        (blas::Uplo) uplo,
        (blas::Op) trans,
        (blas::Diag) diag,
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_Z(B), ldb );
}

} // extern "C"
