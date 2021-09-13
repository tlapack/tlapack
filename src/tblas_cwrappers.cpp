// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <complex.h>
#include "tblas.hpp"

#ifdef BUILD_CBLAS

    #include "tlapack_cblas.h"

    // Mangling
    #ifdef ADD_
        #define BLAS_FUNCTION(fname) cblas_##fname##_
    #else
        #define BLAS_FUNCTION(fname) cblas_##fname
    #endif

    typedef CBLAS_INT   blas_idx_t;
    typedef CBLAS_INT   blas_int_t;
    typedef CBLAS_INDEX blas_iamax_t;

    typedef CBLAS_LAYOUT Layout;
    typedef CBLAS_TRANSPOSE Op;
    typedef CBLAS_UPLO Uplo;
    typedef CBLAS_DIAG Diag;
    typedef CBLAS_SIDE Side;

    // -----------------------------------------------------------------------------
    // Convert CBLAS enum to <T>BLAS enum
    inline blas::Layout toTBLASlayout( Layout layout ) {
        if( layout == CblasRowMajor )      return blas::Layout::RowMajor;
        else if( layout == CblasColMajor ) return blas::Layout::ColMajor;
        else                               return blas::Layout(0);
    }
    inline blas::Op toTBLASop( Op trans ) {
        if( trans == CblasNoTrans )        return blas::Op::NoTrans;
        else if( trans == CblasTrans )     return blas::Op::Trans;
        else if( trans == CblasConjTrans ) return blas::Op::ConjTrans;
        else                            return blas::Op(0);
    }
    inline blas::Uplo toTBLASuplo( Uplo uplo ) {
        if( uplo == CblasUpper )      return blas::Uplo::Upper;
        else if( uplo == CblasLower ) return blas::Uplo::Lower;
        else                          return blas::Uplo(0);
    }
    inline blas::Diag toTBLASdiag( Diag diag ) {
        if( diag == CblasNonUnit )   return blas::Diag::NonUnit;
        else if( diag == CblasUnit ) return blas::Diag::Unit;
        else                         return blas::Diag(0);
    }
    inline blas::Side toTBLASside( Side side ) {
        if( side == CblasLeft )       return blas::Side::Left;
        else if( side == CblasRight ) return blas::Side::Right;
        else                          return blas::Side(0);
    }

#else

    #include "tblas.h"

    // Mangling
    #ifdef ADD_
        #define BLAS_FUNCTION(fname) fname##_
    #else
        #define BLAS_FUNCTION(fname) fname
    #endif

    typedef BLAS_SIZE_T blas_idx_t;
    typedef BLAS_INT_T  blas_int_t;
    typedef BLAS_SIZE_T blas_iamax_t;

    // -----------------------------------------------------------------------------
    // Convert BLAS enum to <T>BLAS enum
    inline blas::Layout toTBLASlayout( Layout layout ) { return (blas::Layout) layout; }
    inline blas::Op toTBLASop( Op op )                 { return (blas::Op) op; }
    inline blas::Uplo toTBLASuplo( Uplo uplo )         { return (blas::Uplo) uplo; }
    inline blas::Diag toTBLASdiag( Diag diag )         { return (blas::Diag) diag; }
    inline blas::Side toTBLASside( Side side )         { return (blas::Side) side; }

#endif

// -----------------------------------------------------------------------------
// Complex types
typedef float  _Complex complexFloat;
typedef double _Complex complexDouble;

typedef std::complex<float> tblas_complexFloat;
typedef std::complex<double> tblas_complexDouble;

#define tblas_cteC(z) reinterpret_cast<const tblas_complexFloat*>( z )
#define tblas_cteZ(z) reinterpret_cast<const tblas_complexDouble*>( z )
#define tblas_C(z) reinterpret_cast<tblas_complexFloat*>( z )
#define tblas_Z(z) reinterpret_cast<tblas_complexDouble*>(  z )
// -----------------------------------------------------------------------------

extern "C" {

#define _sasum BLAS_FUNCTION(sasum)
float _sasum(
    blas_idx_t n,
    float const * x, blas_int_t incx ) {
    return blas::asum<float>(
        n,
        x, incx );
}

#define _dasum BLAS_FUNCTION(dasum)
double _dasum(
    blas_idx_t n,
    double const * x, blas_int_t incx ) {
    return blas::asum<double>(
        n,
        x, incx );
}

#define _casum BLAS_FUNCTION(casum)
float _casum(
    blas_idx_t n,
    complexFloat const * x, blas_int_t incx ) {
    return blas::asum<tblas_complexFloat>(
        n,
        tblas_cteC(x), incx );
}

#define _zasum BLAS_FUNCTION(zasum)
double _zasum(
    blas_idx_t n,
    complexDouble const * x, blas_int_t incx ) {
    return blas::asum<tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx );
}

#define _saxpy BLAS_FUNCTION(saxpy)
void _saxpy(
    blas_idx_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float * y, blas_int_t incy ) {
    return blas::axpy<float, float>(
        n,
        alpha,
        x, incx,
        y, incy );
}

#define _daxpy BLAS_FUNCTION(daxpy)
void _daxpy(
    blas_idx_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double * y, blas_int_t incy ) {
    return blas::axpy<double, double>(
        n,
        alpha,
        x, incx,
        y, incy );
}

#define _caxpy BLAS_FUNCTION(caxpy)
void _caxpy(
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat * y, blas_int_t incy ) {
    return blas::axpy<tblas_complexFloat, tblas_complexFloat>(
        n,
        *tblas_C(&alpha),
        tblas_cteC(x), incx,
        tblas_C(y), incy );
}

#define _zaxpy BLAS_FUNCTION(zaxpy)
void _zaxpy(
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble * y, blas_int_t incy ) {
    return blas::axpy<tblas_complexDouble, tblas_complexDouble>(
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(x), incx,
        tblas_Z(y), incy );
}

#define _scopy BLAS_FUNCTION(scopy)
void _scopy(
    blas_idx_t n,
    float const * x, blas_int_t incx,
    float * y, blas_int_t incy ) {
    return blas::copy<float, float>(
        n,
        x, incx,
        y, incy );
}

#define _dcopy BLAS_FUNCTION(dcopy)
void _dcopy(
    blas_idx_t n,
    double const * x, blas_int_t incx,
    double * y, blas_int_t incy ) {
    return blas::copy<double, double>(
        n,
        x, incx,
        y, incy );
}

#define _ccopy BLAS_FUNCTION(ccopy)
void _ccopy(
    blas_idx_t n,
    complexFloat const * x, blas_int_t incx,
    complexFloat * y, blas_int_t incy ) {
    return blas::copy<tblas_complexFloat, tblas_complexFloat>(
        n,
        tblas_cteC(x), incx,
        tblas_C(y), incy );
}

#define _zcopy BLAS_FUNCTION(zcopy)
void _zcopy(
    blas_idx_t n,
    complexDouble const * x, blas_int_t incx,
    complexDouble * y, blas_int_t incy ) {
    return blas::copy<tblas_complexDouble, tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx,
        tblas_Z(y), incy );
}

#define _sdot BLAS_FUNCTION(sdot)
float _sdot(
    blas_idx_t n,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy ) {
    return blas::dot<float, float>(
        n,
        x, incx,
        y, incy );
}

#define _ddot BLAS_FUNCTION(ddot)
double _ddot(
    blas_idx_t n,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy ) {
    return blas::dot<double, double>(
        n,
        x, incx,
        y, incy );
}

#define _cdot BLAS_FUNCTION(cdot)
complexFloat _cdot(
    blas_idx_t n,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy ) {
    tblas_complexFloat c = blas::dot<tblas_complexFloat, tblas_complexFloat>(
        n,
        tblas_cteC(x), incx,
        tblas_cteC(y), incy );
    return *reinterpret_cast<complexFloat*>(&c);
}

#define _zdot BLAS_FUNCTION(zdot)
complexDouble _zdot(
    blas_idx_t n,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy ) {
    tblas_complexDouble z = blas::dot<tblas_complexDouble, tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy );
    return *reinterpret_cast<complexDouble*>(&z);
}

#define _sdotu BLAS_FUNCTION(sdotu)
float _sdotu(
    blas_idx_t n,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy ) {
    return blas::dotu<float, float>(
        n,
        x, incx,
        y, incy );
}

#define _ddotu BLAS_FUNCTION(ddotu)
double _ddotu(
    blas_idx_t n,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy ) {
    return blas::dotu<double, double>(
        n,
        x, incx,
        y, incy );
}

#define _cdotu BLAS_FUNCTION(cdotu)
complexFloat _cdotu(
    blas_idx_t n,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy ) {
    tblas_complexFloat c = blas::dotu<tblas_complexFloat, tblas_complexFloat>(
        n,
        tblas_cteC(x), incx,
        tblas_cteC(y), incy );
    return *reinterpret_cast<complexFloat*>(&c);
}

#define _zdotu BLAS_FUNCTION(zdotu)
complexDouble _zdotu(
    blas_idx_t n,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy ) {
    tblas_complexDouble z = blas::dotu<tblas_complexDouble, tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy );
    return *reinterpret_cast<complexDouble*>(&z);
}

#define _isamax BLAS_FUNCTION(isamax)
blas_iamax_t _isamax(
    blas_idx_t n,
    float const * x, blas_int_t incx ) {
    return (blas_iamax_t) blas::iamax<float>(
        n,
        x, incx );
}

#define _idamax BLAS_FUNCTION(idamax)
blas_iamax_t _idamax(
    blas_idx_t n,
    double const * x, blas_int_t incx ) {
    return (blas_iamax_t) blas::iamax<double>(
        n,
        x, incx );
}

#define _icamax BLAS_FUNCTION(icamax)
blas_iamax_t _icamax(
    blas_idx_t n,
    complexFloat const * x, blas_int_t incx ) {
    return (blas_iamax_t) blas::iamax<tblas_complexFloat>(
        n,
        tblas_cteC(x), incx );
}

#define _izamax BLAS_FUNCTION(izamax)
blas_iamax_t _izamax(
    blas_idx_t n,
    complexDouble const * x, blas_int_t incx ) {
    return (blas_iamax_t) blas::iamax<tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx );
}

#define _snrm2 BLAS_FUNCTION(snrm2)
float _snrm2(
    blas_idx_t n,
    float const * x, blas_int_t incx ) {
    return blas::nrm2<float>(
        n,
        x, incx );
}

#define _dnrm2 BLAS_FUNCTION(dnrm2)
double _dnrm2(
    blas_idx_t n,
    double const * x, blas_int_t incx ) {
    return blas::nrm2<double>(
        n,
        x, incx );
}

#define _cnrm2 BLAS_FUNCTION(cnrm2)
float _cnrm2(
    blas_idx_t n,
    complexFloat const * x, blas_int_t incx ) {
    return blas::nrm2<tblas_complexFloat>(
        n,
        tblas_cteC(x), incx );
}

#define _znrm2 BLAS_FUNCTION(znrm2)
double _znrm2(
    blas_idx_t n,
    complexDouble const * x, blas_int_t incx ) {
    return blas::nrm2<tblas_complexDouble>(
        n,
        tblas_cteZ(x), incx );
}

#define _srot BLAS_FUNCTION(srot)
void _srot(
    blas_idx_t n,
    float * x, blas_int_t incx,
    float * y, blas_int_t incy,
    float c,
    float s ) {
    return blas::rot<float, float>(
        n,
        x, incx,
        y, incy,
        c,
        s );
}

#define _drot BLAS_FUNCTION(drot)
void _drot(
    blas_idx_t n,
    double * x, blas_int_t incx,
    double * y, blas_int_t incy,
    double c,
    double s ) {
    return blas::rot<double, double>(
        n,
        x, incx,
        y, incy,
        c,
        s );
}

#define _csrot BLAS_FUNCTION(csrot)
void _csrot(
    blas_idx_t n,
    complexFloat * x, blas_int_t incx,
    complexFloat * y, blas_int_t incy,
    float c,
    float s ) {
    return blas::rot(
        n,
        tblas_C(x), incx,
        tblas_C(y), incy,
        c,
        s );
}

#define _zdrot BLAS_FUNCTION(zdrot)
void _zdrot(
    blas_idx_t n,
    complexDouble * x, blas_int_t incx,
    complexDouble * y, blas_int_t incy,
    double c,
    double s ) {
    return blas::rot(
        n,
        tblas_Z(x), incx,
        tblas_Z(y), incy,
        c,
        s );
}

#define _crot BLAS_FUNCTION(crot)
void _crot(
    blas_idx_t n,
    complexFloat * x, blas_int_t incx,
    complexFloat * y, blas_int_t incy,
    float c,
    complexFloat s ) {
    return blas::rot<tblas_complexFloat, tblas_complexFloat>(
        n,
        tblas_C(x), incx,
        tblas_C(y), incy,
        c,
        *tblas_C(&s) );
}

#define _zrot BLAS_FUNCTION(zrot)
void _zrot(
    blas_idx_t n,
    complexDouble * x, blas_int_t incx,
    complexDouble * y, blas_int_t incy,
    double c,
    complexDouble s ) {
    return blas::rot<tblas_complexDouble, tblas_complexDouble>(
        n,
        tblas_Z(x), incx,
        tblas_Z(y), incy,
        c,
        *tblas_Z(&s) );
}

#define _srotg BLAS_FUNCTION(srotg)
void _srotg(
    float * a,
    float * b,
    float * c,
    float * s ) {
    return blas::rotg<float>(
        a,
        b,
        c,
        s );
}

#define _drotg BLAS_FUNCTION(drotg)
void _drotg(
    double * a,
    double * b,
    double * c,
    double * s ) {
    return blas::rotg<double>(
        a,
        b,
        c,
        s );
}

#define _crotg BLAS_FUNCTION(crotg)
void _crotg(
    complexFloat * a,
    complexFloat * b,
    float * c,
    complexFloat * s ) {
    return blas::rotg<tblas_complexFloat>(
        tblas_C(a),
        tblas_C(b),
        c,
        tblas_C(s) );
}

#define _zrotg BLAS_FUNCTION(zrotg)
void _zrotg(
    complexDouble * a,
    complexDouble * b,
    double * c,
    complexDouble * s ) {
    return blas::rotg<tblas_complexDouble>(
        tblas_Z(a),
        tblas_Z(b),
        c,
        tblas_Z(s) );
}

#define _srotm BLAS_FUNCTION(srotm)
void _srotm(
    blas_idx_t n,
    float * x, blas_int_t incx,
    float * y, blas_int_t incy,
    float const * param ) {
    return blas::rotm<float, float>(
        n,
        x, incx,
        y, incy,
        param );
}

#define _drotm BLAS_FUNCTION(drotm)
void _drotm(
    blas_idx_t n,
    double * x, blas_int_t incx,
    double * y, blas_int_t incy,
    double const * param ) {
    return blas::rotm<double, double>(
        n,
        x, incx,
        y, incy,
        param );
}

#define _srotmg BLAS_FUNCTION(srotmg)
void _srotmg(
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

#define _drotmg BLAS_FUNCTION(drotmg)
void _drotmg(
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

#define _sscal BLAS_FUNCTION(sscal)
void _sscal(
    blas_idx_t n,
    float alpha,
    float * x, blas_int_t incx ) {
    return blas::scal<float>(
        n,
        alpha,
        x, incx );
}

#define _dscal BLAS_FUNCTION(dscal)
void _dscal(
    blas_idx_t n,
    double alpha,
    double * x, blas_int_t incx ) {
    return blas::scal<double>(
        n,
        alpha,
        x, incx );
}

#define _cscal BLAS_FUNCTION(cscal)
void _cscal(
    blas_idx_t n,
    complexFloat alpha,
    complexFloat * x, blas_int_t incx ) {
    return blas::scal<tblas_complexFloat>(
        n,
        *tblas_C(&alpha),
        tblas_C(x), incx );
}

#define _zscal BLAS_FUNCTION(zscal)
void _zscal(
    blas_idx_t n,
    complexDouble alpha,
    complexDouble * x, blas_int_t incx ) {
    return blas::scal<tblas_complexDouble>(
        n,
        *tblas_Z(&alpha),
        tblas_Z(x), incx );
}

#define _sswap BLAS_FUNCTION(sswap)
void _sswap(
    blas_idx_t n,
    float * x, blas_int_t incx,
    float * y, blas_int_t incy ) {
    return blas::swap<float, float>(
        n,
        x, incx,
        y, incy );
}

#define _dswap BLAS_FUNCTION(dswap)
void _dswap(
    blas_idx_t n,
    double * x, blas_int_t incx,
    double * y, blas_int_t incy ) {
    return blas::swap<double, double>(
        n,
        x, incx,
        y, incy );
}

#define _cswap BLAS_FUNCTION(cswap)
void _cswap(
    blas_idx_t n,
    complexFloat * x, blas_int_t incx,
    complexFloat * y, blas_int_t incy ) {
    return blas::swap<tblas_complexFloat, tblas_complexFloat>(
        n,
        tblas_C(x), incx,
        tblas_C(y), incy );
}

#define _zswap BLAS_FUNCTION(zswap)
void _zswap(
    blas_idx_t n,
    complexDouble * x, blas_int_t incx,
    complexDouble * y, blas_int_t incy ) {
    return blas::swap<tblas_complexDouble, tblas_complexDouble>(
        n,
        tblas_Z(x), incx,
        tblas_Z(y), incy );
}

#define _sgemv BLAS_FUNCTION(sgemv)
void _sgemv(
    Layout layout,
    Op trans,
    blas_idx_t m,
    blas_idx_t n,
    float alpha,
    float const * A, blas_idx_t lda,
    float const * x, blas_int_t incx,
    float beta,
    float * y, blas_int_t incy ) {
    return blas::gemv<float, float, float>(
        toTBLASlayout( layout ),
        toTBLASop( trans ),
        m,
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

#define _dgemv BLAS_FUNCTION(dgemv)
void _dgemv(
    Layout layout,
    Op trans,
    blas_idx_t m,
    blas_idx_t n,
    double alpha,
    double const * A, blas_idx_t lda,
    double const * x, blas_int_t incx,
    double beta,
    double * y, blas_int_t incy ) {
    return blas::gemv<double, double, double>(
        toTBLASlayout( layout ),
        toTBLASop( trans ),
        m,
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

#define _cgemv BLAS_FUNCTION(cgemv)
void _cgemv(
    Layout layout,
    Op trans,
    blas_idx_t m,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat const * x, blas_int_t incx,
    complexFloat beta,
    complexFloat * y, blas_int_t incy ) {
    return blas::gemv<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASop( trans ),
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(x), incx,
        *tblas_C(&beta),
        tblas_C(y), incy );
}

#define _zgemv BLAS_FUNCTION(zgemv)
void _zgemv(
    Layout layout,
    Op trans,
    blas_idx_t m,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble const * x, blas_int_t incx,
    complexDouble beta,
    complexDouble * y, blas_int_t incy ) {
    return blas::gemv<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASop( trans ),
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(x), incx,
        *tblas_Z(&beta),
        tblas_Z(y), incy );
}

#define _sger BLAS_FUNCTION(sger)
void _sger(
    Layout layout,
    blas_idx_t m,
    blas_idx_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy,
    float * A, blas_idx_t lda ) {
    return blas::ger<float, float, float>(
        toTBLASlayout( layout ),
        m,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

#define _dger BLAS_FUNCTION(dger)
void _dger(
    Layout layout,
    blas_idx_t m,
    blas_idx_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy,
    double * A, blas_idx_t lda ) {
    return blas::ger<double, double, double>(
        toTBLASlayout( layout ),
        m,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

#define _cger BLAS_FUNCTION(cger)
void _cger(
    Layout layout,
    blas_idx_t m,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy,
    complexFloat * A, blas_idx_t lda ) {
    return blas::ger<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(x), incx,
        tblas_cteC(y), incy,
        tblas_C(A), lda );
}

#define _zger BLAS_FUNCTION(zger)
void _zger(
    Layout layout,
    blas_idx_t m,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy,
    complexDouble * A, blas_idx_t lda ) {
    return blas::ger<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy,
        tblas_Z(A), lda );
}

#define _sgeru BLAS_FUNCTION(sgeru)
void _sgeru(
    Layout layout,
    blas_idx_t m,
    blas_idx_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy,
    float * A, blas_idx_t lda ) {
    return blas::geru<float, float, float>(
        toTBLASlayout( layout ),
        m,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

#define _dgeru BLAS_FUNCTION(dgeru)
void _dgeru(
    Layout layout,
    blas_idx_t m,
    blas_idx_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy,
    double * A, blas_idx_t lda ) {
    return blas::geru<double, double, double>(
        toTBLASlayout( layout ),
        m,
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

#define _cgeru BLAS_FUNCTION(cgeru)
void _cgeru(
    Layout layout,
    blas_idx_t m,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy,
    complexFloat * A, blas_idx_t lda ) {
    return blas::geru<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(x), incx,
        tblas_cteC(y), incy,
        tblas_C(A), lda );
}

#define _zgeru BLAS_FUNCTION(zgeru)
void _zgeru(
    Layout layout,
    blas_idx_t m,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy,
    complexDouble * A, blas_idx_t lda ) {
    return blas::geru<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy,
        tblas_Z(A), lda );
}

#define _shemv BLAS_FUNCTION(shemv)
void _shemv(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    float alpha,
    float const * A, blas_idx_t lda,
    float const * x, blas_int_t incx,
    float beta,
    float * y, blas_int_t incy ) {
    return blas::hemv<float, float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

#define _dhemv BLAS_FUNCTION(dhemv)
void _dhemv(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    double alpha,
    double const * A, blas_idx_t lda,
    double const * x, blas_int_t incx,
    double beta,
    double * y, blas_int_t incy ) {
    return blas::hemv<double, double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

#define _chemv BLAS_FUNCTION(chemv)
void _chemv(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat const * x, blas_int_t incx,
    complexFloat beta,
    complexFloat * y, blas_int_t incy ) {
    return blas::hemv<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(x), incx,
        *tblas_C(&beta),
        tblas_C(y), incy );
}

#define _zhemv BLAS_FUNCTION(zhemv)
void _zhemv(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble const * x, blas_int_t incx,
    complexDouble beta,
    complexDouble * y, blas_int_t incy ) {
    return blas::hemv<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(x), incx,
        *tblas_Z(&beta),
        tblas_Z(y), incy );
}

#define _sher BLAS_FUNCTION(sher)
void _sher(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float * A, blas_idx_t lda ) {
    return blas::her<float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        x, incx,
        A, lda );
}

#define _dher BLAS_FUNCTION(dher)
void _dher(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double * A, blas_idx_t lda ) {
    return blas::her<double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        x, incx,
        A, lda );
}

#define _cher BLAS_FUNCTION(cher)
void _cher(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    float alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat * A, blas_idx_t lda ) {
    return blas::her<tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        tblas_cteC(x), incx,
        tblas_C(A), lda );
}

#define _zher BLAS_FUNCTION(zher)
void _zher(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    double alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble * A, blas_idx_t lda ) {
    return blas::her<tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        tblas_cteZ(x), incx,
        tblas_Z(A), lda );
}

#define _sher2 BLAS_FUNCTION(sher2)
void _sher2(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy,
    float * A, blas_idx_t lda ) {
    return blas::her2<float, float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

#define _dher2 BLAS_FUNCTION(dher2)
void _dher2(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy,
    double * A, blas_idx_t lda ) {
    return blas::her2<double, double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

#define _cher2 BLAS_FUNCTION(cher2)
void _cher2(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy,
    complexFloat * A, blas_idx_t lda ) {
    return blas::her2<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        *tblas_C(&alpha),
        tblas_cteC(x), incx,
        tblas_cteC(y), incy,
        tblas_C(A), lda );
}

#define _zher2 BLAS_FUNCTION(zher2)
void _zher2(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy,
    complexDouble * A, blas_idx_t lda ) {
    return blas::her2<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy,
        tblas_Z(A), lda );
}

#define _ssymv BLAS_FUNCTION(ssymv)
void _ssymv(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    float alpha,
    float const * A, blas_idx_t lda,
    float const * x, blas_int_t incx,
    float beta,
    float * y, blas_int_t incy ) {
    return blas::symv<float, float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

#define _dsymv BLAS_FUNCTION(dsymv)
void _dsymv(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    double alpha,
    double const * A, blas_idx_t lda,
    double const * x, blas_int_t incx,
    double beta,
    double * y, blas_int_t incy ) {
    return blas::symv<double, double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        A, lda,
        x, incx,
        beta,
        y, incy );
}

#define _csymv BLAS_FUNCTION(csymv)
void _csymv(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat const * x, blas_int_t incx,
    complexFloat beta,
    complexFloat * y, blas_int_t incy ) {
    return blas::symv<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(x), incx,
        *tblas_C(&beta),
        tblas_C(y), incy );
}

#define _zsymv BLAS_FUNCTION(zsymv)
void _zsymv(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble const * x, blas_int_t incx,
    complexDouble beta,
    complexDouble * y, blas_int_t incy ) {
    return blas::symv<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(x), incx,
        *tblas_Z(&beta),
        tblas_Z(y), incy );
}

#define _ssyr BLAS_FUNCTION(ssyr)
void _ssyr(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float * A, blas_idx_t lda ) {
    return blas::syr<float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        x, incx,
        A, lda );
}

#define _dsyr BLAS_FUNCTION(dsyr)
void _dsyr(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double * A, blas_idx_t lda ) {
    return blas::syr<double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        x, incx,
        A, lda );
}

#define _ssyr2 BLAS_FUNCTION(ssyr2)
void _ssyr2(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    float alpha,
    float const * x, blas_int_t incx,
    float const * y, blas_int_t incy,
    float * A, blas_idx_t lda ) {
    return blas::syr2<float, float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

#define _dsyr2 BLAS_FUNCTION(dsyr2)
void _dsyr2(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    double alpha,
    double const * x, blas_int_t incx,
    double const * y, blas_int_t incy,
    double * A, blas_idx_t lda ) {
    return blas::syr2<double, double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        alpha,
        x, incx,
        y, incy,
        A, lda );
}

#define _csyr2 BLAS_FUNCTION(csyr2)
void _csyr2(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * x, blas_int_t incx,
    complexFloat const * y, blas_int_t incy,
    complexFloat * A, blas_idx_t lda ) {
    return blas::syr2<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        *tblas_C(&alpha),
        tblas_cteC(x), incx,
        tblas_cteC(y), incy,
        tblas_C(A), lda );
}

#define _zsyr2 BLAS_FUNCTION(zsyr2)
void _zsyr2(
    Layout layout,
    Uplo uplo,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * x, blas_int_t incx,
    complexDouble const * y, blas_int_t incy,
    complexDouble * A, blas_idx_t lda ) {
    return blas::syr2<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(x), incx,
        tblas_cteZ(y), incy,
        tblas_Z(A), lda );
}

#define _strmv BLAS_FUNCTION(strmv)
void _strmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t n,
    float const * A, blas_idx_t lda,
    float * x, blas_int_t incx ) {
    return blas::trmv<float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        n,
        A, lda,
        x, incx );
}

#define _dtrmv BLAS_FUNCTION(dtrmv)
void _dtrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t n,
    double const * A, blas_idx_t lda,
    double * x, blas_int_t incx ) {
    return blas::trmv<double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        n,
        A, lda,
        x, incx );
}

#define _ctrmv BLAS_FUNCTION(ctrmv)
void _ctrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t n,
    complexFloat const * A, blas_idx_t lda,
    complexFloat * x, blas_int_t incx ) {
    return blas::trmv<tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        n,
        tblas_cteC(A), lda,
        tblas_C(x), incx );
}

#define _ztrmv BLAS_FUNCTION(ztrmv)
void _ztrmv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t n,
    complexDouble const * A, blas_idx_t lda,
    complexDouble * x, blas_int_t incx ) {
    return blas::trmv<tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        n,
        tblas_cteZ(A), lda,
        tblas_Z(x), incx );
}

#define _strsv BLAS_FUNCTION(strsv)
void _strsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t n,
    float const * A, blas_idx_t lda,
    float * x, blas_int_t incx ) {
    return blas::trsv<float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        n,
        A, lda,
        x, incx );
}

#define _dtrsv BLAS_FUNCTION(dtrsv)
void _dtrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t n,
    double const * A, blas_idx_t lda,
    double * x, blas_int_t incx ) {
    return blas::trsv<double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        n,
        A, lda,
        x, incx );
}

#define _ctrsv BLAS_FUNCTION(ctrsv)
void _ctrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t n,
    complexFloat const * A, blas_idx_t lda,
    complexFloat * x, blas_int_t incx ) {
    return blas::trsv<tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        n,
        tblas_cteC(A), lda,
        tblas_C(x), incx );
}

#define _ztrsv BLAS_FUNCTION(ztrsv)
void _ztrsv(
    Layout layout,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t n,
    complexDouble const * A, blas_idx_t lda,
    complexDouble * x, blas_int_t incx ) {
    return blas::trsv<tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        n,
        tblas_cteZ(A), lda,
        tblas_Z(x), incx );
}

#define _sgemm BLAS_FUNCTION(sgemm)
void _sgemm(
    Layout layout,
    Op transA,
    Op transB,
    blas_idx_t m,
    blas_idx_t n,
    blas_idx_t k,
    float alpha,
    float const * A, blas_idx_t lda,
    float const * B, blas_idx_t ldb,
    float beta,
    float * C, blas_idx_t ldc ) {
    return blas::gemm<float, float, float>(
        toTBLASlayout( layout ),
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

#define _dgemm BLAS_FUNCTION(dgemm)
void _dgemm(
    Layout layout,
    Op transA,
    Op transB,
    blas_idx_t m,
    blas_idx_t n,
    blas_idx_t k,
    double alpha,
    double const * A, blas_idx_t lda,
    double const * B, blas_idx_t ldb,
    double beta,
    double * C, blas_idx_t ldc ) {
    return blas::gemm<double, double, double>(
        toTBLASlayout( layout ),
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

#define _cgemm BLAS_FUNCTION(cgemm)
void _cgemm(
    Layout layout,
    Op transA,
    Op transB,
    blas_idx_t m,
    blas_idx_t n,
    blas_idx_t k,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat const * B, blas_idx_t ldb,
    complexFloat beta,
    complexFloat * C, blas_idx_t ldc ) {
    return blas::gemm<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
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

#define _zgemm BLAS_FUNCTION(zgemm)
void _zgemm(
    Layout layout,
    Op transA,
    Op transB,
    blas_idx_t m,
    blas_idx_t n,
    blas_idx_t k,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble const * B, blas_idx_t ldb,
    complexDouble beta,
    complexDouble * C, blas_idx_t ldc ) {
    return blas::gemm<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
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

#define _shemm BLAS_FUNCTION(shemm)
void _shemm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_idx_t m,
    blas_idx_t n,
    float alpha,
    float const * A, blas_idx_t lda,
    float const * B, blas_idx_t ldb,
    float beta,
    float * C, blas_idx_t ldc ) {
    return blas::hemm<float, float, float>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        m,
        n,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

#define _dhemm BLAS_FUNCTION(dhemm)
void _dhemm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_idx_t m,
    blas_idx_t n,
    double alpha,
    double const * A, blas_idx_t lda,
    double const * B, blas_idx_t ldb,
    double beta,
    double * C, blas_idx_t ldc ) {
    return blas::hemm<double, double, double>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        m,
        n,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

#define _chemm BLAS_FUNCTION(chemm)
void _chemm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_idx_t m,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat const * B, blas_idx_t ldb,
    complexFloat beta,
    complexFloat * C, blas_idx_t ldc ) {
    return blas::hemm<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(B), ldb,
        *tblas_C(&beta),
        tblas_C(C), ldc );
}

#define _zhemm BLAS_FUNCTION(zhemm)
void _zhemm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_idx_t m,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble const * B, blas_idx_t ldb,
    complexDouble beta,
    complexDouble * C, blas_idx_t ldc ) {
    return blas::hemm<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(B), ldb,
        *tblas_Z(&beta),
        tblas_Z(C), ldc );
}

#define _sher2k BLAS_FUNCTION(sher2k)
void _sher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    float alpha,
    float const * A, blas_idx_t lda,
    float const * B, blas_idx_t ldb,
    float beta,
    float * C, blas_idx_t ldc ) {
    return blas::her2k<float, float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

#define _dher2k BLAS_FUNCTION(dher2k)
void _dher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    double alpha,
    double const * A, blas_idx_t lda,
    double const * B, blas_idx_t ldb,
    double beta,
    double * C, blas_idx_t ldc ) {
    return blas::her2k<double, double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

#define _cher2k BLAS_FUNCTION(cher2k)
void _cher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat const * B, blas_idx_t ldb,
    float beta,
    complexFloat * C, blas_idx_t ldc ) {
    return blas::her2k<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(B), ldb,
        beta,
        tblas_C(C), ldc );
}

#define _zher2k BLAS_FUNCTION(zher2k)
void _zher2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble const * B, blas_idx_t ldb,
    double beta,
    complexDouble * C, blas_idx_t ldc ) {
    return blas::her2k<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(B), ldb,
        beta,
        tblas_Z(C), ldc );
}

#define _sherk BLAS_FUNCTION(sherk)
void _sherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    float alpha,
    float const * A, blas_idx_t lda,
    float beta,
    float * C, blas_idx_t ldc ) {
    return blas::herk<float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        alpha,
        A, lda,
        beta,
        C, ldc );
}

#define _dherk BLAS_FUNCTION(dherk)
void _dherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    double alpha,
    double const * A, blas_idx_t lda,
    double beta,
    double * C, blas_idx_t ldc ) {
    return blas::herk<double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        alpha,
        A, lda,
        beta,
        C, ldc );
}

#define _cherk BLAS_FUNCTION(cherk)
void _cherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    float alpha,
    complexFloat const * A, blas_idx_t lda,
    float beta,
    complexFloat * C, blas_idx_t ldc ) {
    return blas::herk<tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        alpha,
        tblas_cteC(A), lda,
        beta,
        tblas_C(C), ldc );
}

#define _zherk BLAS_FUNCTION(zherk)
void _zherk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    double alpha,
    complexDouble const * A, blas_idx_t lda,
    double beta,
    complexDouble * C, blas_idx_t ldc ) {
    return blas::herk<tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        alpha,
        tblas_cteZ(A), lda,
        beta,
        tblas_Z(C), ldc );
}

#define _ssymm BLAS_FUNCTION(ssymm)
void _ssymm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_idx_t m,
    blas_idx_t n,
    float alpha,
    float const * A, blas_idx_t lda,
    float const * B, blas_idx_t ldb,
    float beta,
    float * C, blas_idx_t ldc ) {
    return blas::symm<float, float, float>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        m,
        n,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

#define _dsymm BLAS_FUNCTION(dsymm)
void _dsymm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_idx_t m,
    blas_idx_t n,
    double alpha,
    double const * A, blas_idx_t lda,
    double const * B, blas_idx_t ldb,
    double beta,
    double * C, blas_idx_t ldc ) {
    return blas::symm<double, double, double>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        m,
        n,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

#define _csymm BLAS_FUNCTION(csymm)
void _csymm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_idx_t m,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat const * B, blas_idx_t ldb,
    complexFloat beta,
    complexFloat * C, blas_idx_t ldc ) {
    return blas::symm<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(B), ldb,
        *tblas_C(&beta),
        tblas_C(C), ldc );
}

#define _zsymm BLAS_FUNCTION(zsymm)
void _zsymm(
    Layout layout,
    Side side,
    Uplo uplo,
    blas_idx_t m,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble const * B, blas_idx_t ldb,
    complexDouble beta,
    complexDouble * C, blas_idx_t ldc ) {
    return blas::symm<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(B), ldb,
        *tblas_Z(&beta),
        tblas_Z(C), ldc );
}

#define _ssyr2k BLAS_FUNCTION(ssyr2k)
void _ssyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    float alpha,
    float const * A, blas_idx_t lda,
    float const * B, blas_idx_t ldb,
    float beta,
    float * C, blas_idx_t ldc ) {
    return blas::syr2k<float, float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

#define _dsyr2k BLAS_FUNCTION(dsyr2k)
void _dsyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    double alpha,
    double const * A, blas_idx_t lda,
    double const * B, blas_idx_t ldb,
    double beta,
    double * C, blas_idx_t ldc ) {
    return blas::syr2k<double, double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc );
}

#define _csyr2k BLAS_FUNCTION(csyr2k)
void _csyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat const * B, blas_idx_t ldb,
    complexFloat beta,
    complexFloat * C, blas_idx_t ldc ) {
    return blas::syr2k<tblas_complexFloat, tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_cteC(B), ldb,
        *tblas_C(&beta),
        tblas_C(C), ldc );
}

#define _zsyr2k BLAS_FUNCTION(zsyr2k)
void _zsyr2k(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble const * B, blas_idx_t ldb,
    complexDouble beta,
    complexDouble * C, blas_idx_t ldc ) {
    return blas::syr2k<tblas_complexDouble, tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_cteZ(B), ldb,
        *tblas_Z(&beta),
        tblas_Z(C), ldc );
}

#define _ssyrk BLAS_FUNCTION(ssyrk)
void _ssyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    float alpha,
    float const * A, blas_idx_t lda,
    float beta,
    float * C, blas_idx_t ldc ) {
    return blas::syrk<float, float>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        alpha,
        A, lda,
        beta,
        C, ldc );
}

#define _dsyrk BLAS_FUNCTION(dsyrk)
void _dsyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    double alpha,
    double const * A, blas_idx_t lda,
    double beta,
    double * C, blas_idx_t ldc ) {
    return blas::syrk<double, double>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        alpha,
        A, lda,
        beta,
        C, ldc );
}

#define _csyrk BLAS_FUNCTION(csyrk)
void _csyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat beta,
    complexFloat * C, blas_idx_t ldc ) {
    return blas::syrk<tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        *tblas_C(&beta),
        tblas_C(C), ldc );
}

#define _zsyrk BLAS_FUNCTION(zsyrk)
void _zsyrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    blas_idx_t n,
    blas_idx_t k,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble beta,
    complexDouble * C, blas_idx_t ldc ) {
    return blas::syrk<tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        n,
        k,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        *tblas_Z(&beta),
        tblas_Z(C), ldc );
}

#define _strmm BLAS_FUNCTION(strmm)
void _strmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t m,
    blas_idx_t n,
    float alpha,
    float const * A, blas_idx_t lda,
    float * B, blas_idx_t ldb ) {
    return blas::trmm<float, float>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        m,
        n,
        alpha,
        A, lda,
        B, ldb );
}

#define _dtrmm BLAS_FUNCTION(dtrmm)
void _dtrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t m,
    blas_idx_t n,
    double alpha,
    double const * A, blas_idx_t lda,
    double * B, blas_idx_t ldb ) {
    return blas::trmm<double, double>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        m,
        n,
        alpha,
        A, lda,
        B, ldb );
}

#define _ctrmm BLAS_FUNCTION(ctrmm)
void _ctrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t m,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat * B, blas_idx_t ldb ) {
    return blas::trmm<tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_C(B), ldb );
}

#define _ztrmm BLAS_FUNCTION(ztrmm)
void _ztrmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t m,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble * B, blas_idx_t ldb ) {
    return blas::trmm<tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_Z(B), ldb );
}

#define _strsm BLAS_FUNCTION(strsm)
void _strsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t m,
    blas_idx_t n,
    float alpha,
    float const * A, blas_idx_t lda,
    float * B, blas_idx_t ldb ) {
    return blas::trsm<float, float>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        m,
        n,
        alpha,
        A, lda,
        B, ldb );
}

#define _dtrsm BLAS_FUNCTION(dtrsm)
void _dtrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t m,
    blas_idx_t n,
    double alpha,
    double const * A, blas_idx_t lda,
    double * B, blas_idx_t ldb ) {
    return blas::trsm<double, double>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        m,
        n,
        alpha,
        A, lda,
        B, ldb );
}

#define _ctrsm BLAS_FUNCTION(ctrsm)
void _ctrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t m,
    blas_idx_t n,
    complexFloat alpha,
    complexFloat const * A, blas_idx_t lda,
    complexFloat * B, blas_idx_t ldb ) {
    return blas::trsm<tblas_complexFloat, tblas_complexFloat>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        m,
        n,
        *tblas_C(&alpha),
        tblas_cteC(A), lda,
        tblas_C(B), ldb );
}

#define _ztrsm BLAS_FUNCTION(ztrsm)
void _ztrsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    blas_idx_t m,
    blas_idx_t n,
    complexDouble alpha,
    complexDouble const * A, blas_idx_t lda,
    complexDouble * B, blas_idx_t ldb ) {
    return blas::trsm<tblas_complexDouble, tblas_complexDouble>(
        toTBLASlayout( layout ),
        toTBLASside( side ),
        toTBLASuplo( uplo ),
        toTBLASop( trans ),
        toTBLASdiag( diag ),
        m,
        n,
        *tblas_Z(&alpha),
        tblas_cteZ(A), lda,
        tblas_Z(B), ldb );
}

} // extern "C"
