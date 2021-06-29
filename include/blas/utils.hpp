// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_UTILS_HH__
#define __TBLAS_UTILS_HH__

#include "blas/types.hpp"
#include <limits>
#include <exception>
#include <string>
#include <cstdarg>
#include <cmath>

#ifdef USE_MPFR
    #include <mpreal.h>
#endif

namespace blas {

// -----------------------------------------------------------------------------
// Use routines from std C++
using std::real;
using std::imag;
using std::abs; // Contains the 2-norm for the complex case
using std::isinf;
using std::isnan;
using std::ceil;
using std::floor;

// -----------------------------------------------------------------------------
// Use MPFR interface
#ifdef USE_MPFR
    inline mpfr::mpreal real( const mpfr::mpreal& x ) { return x; }
    inline mpfr::mpreal imag( const mpfr::mpreal& x ) { return 0; }
    using mpfr::abs;
    using mpfr::isinf;
    using mpfr::isnan;
    using mpfr::ceil;
    using mpfr::floor;
#endif

/** Extend conj to real datatypes.
 * 
 * @usage:
 *     using blas::conj;
 *     scalar_t x = ...
 *     scalar_t y = conj( x );
 * 
 * @param[in] x Real number
 * @return x
 * 
 * @note C++11 returns complex<real_t> instead of real_t. @see std::conj
 * 
 * @ingroup utils
 */
template< typename real_t >
inline real_t conj( const real_t& x )
{
    // This prohibits complex types; it can't be called as y = blas::conj( x ).
    static_assert( ! is_complex<real_t>::value,
                    "Usage: using blas::conj; y = conj(x); NOT: y = blas::conj(x);" );
    return x;
}

// -----------------------------------------------------------------------------
// max that works with different data types
// and any number of arguments: max( a, b, c, d )

// one argument
template< typename T >
inline T max( const T& x ) { return x; }

// two arguments
template< typename T1, typename T2 >
inline scalar_type< T1, T2 >
    max( const T1& x, const T2& y )
{
    return (x >= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
inline scalar_type< T1, Types... >
    max( const T1& first, const Types&... args )
{
    return max( first, max( args... ) );
}

// -----------------------------------------------------------------------------
// min that works with different data types
// and any number of arguments: min( a, b, c, d )

// one argument
template< typename T >
inline T min( const T& x ) { return x; }

// two arguments
template< typename T1, typename T2 >
inline scalar_type< T1, T2 >
    min( const T1& x, const T2& y )
{
    return (x <= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
inline scalar_type< T1, Types... >
    min( const T1& first, const Types&... args )
{
    return min( first, min( args... ) );
}

// -----------------------------------------------------------------------------
// Generate a scalar from real and imaginary parts.
// For real scalars, the imaginary part is ignored.

// For real scalar types.
template <typename real_t>
struct MakeScalarTraits {
    static inline real_t make( const real_t& re, const real_t& im )
        { return re; }
};

// For complex scalar types.
template <typename real_t>
struct MakeScalarTraits< std::complex<real_t> > {
    static inline std::complex<real_t> make( const real_t& re, const real_t& im )
        { return std::complex<real_t>( re, im ); }
};

template <typename scalar_t>
inline scalar_t make_scalar( blas::real_type<scalar_t> re,
                             blas::real_type<scalar_t> im=0 )
{
    return MakeScalarTraits<scalar_t>::make( re, im );
}

// -----------------------------------------------------------------------------
/// Type-safe sgn function
/// @see Source: https://stackoverflow.com/a/4609795/5253097
///
template <typename real_t>
inline int sgn( const real_t& val ) {
    return (real_t(0) < val) - (val < real_t(0));
}

#ifdef USE_MPFR
    template<> 
    inline int sgn( const mpfr::mpreal& x )
    { return mpfr::sgn( x ); }
#endif

// -----------------------------------------------------------------------------
/// sqrt, needed because std C++11 template returns double.
/// Note that the template in std::complex return the desired std::complex<T>.
template< typename T >
inline T sqrt( const T& x )
{ return std::sqrt( x ); }

#ifdef USE_MPFR
    template<> 
    inline mpfr::mpreal sqrt( const mpfr::mpreal& x )
    { return mpfr::sqrt( x ); }
#endif

// -----------------------------------------------------------------------------
/// pow, avoids promotion to double from std C++11.
/// Note that the template in std::complex return the desired std::complex<T>.
template< typename T >
inline T pow( const T& base, const T& exp )
{ return std::pow( base, exp ); }

template< typename T >
inline T pow( const int base, const T& exp )
{ return std::pow( (double)base, exp ); }

#ifdef USE_MPFR
    template<>
    inline mpfr::mpreal pow(const mpfr::mpreal& a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const mpfr::mpreal& a, const unsigned int b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const mpfr::mpreal& a, const int b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const mpfr::mpreal& a, const long double b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const mpfr::mpreal& a, const double b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const unsigned int a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const long int a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
    template<>
    inline mpfr::mpreal pow(const int a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const long double a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const double a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
#endif

// -----------------------------------------------------------------------------
/// 1-norm absolute value, |Re(x)| + |Im(x)|
template< typename T >
inline T abs1( T x ) { return abs( x ); }

template< typename T >
inline T abs1( std::complex<T> x )
{
    return abs( real(x) ) + abs( imag(x) );
}

// -----------------------------------------------------------------------------
/// isnan for complex numbers
template< typename T >
inline bool isnan( const std::complex<T>& x )
{
    return isnan( real(x) ) || isnan( imag(x) );
}

// -----------------------------------------------------------------------------
/// isinf for complex numbers
template< typename T >
inline bool isinf( std::complex<T> x )
{
    return isinf( real(x) ) || isinf( imag(x) );
}

// -----------------------------------------------------------------------------
/// Exception class for BLAS errors.
class Error: public std::exception {
public:
    /// Constructs BLAS error
    Error():
        std::exception()
    {}

    /// Constructs BLAS error with message
    Error( std::string const& msg ):
        std::exception(),
        msg_( msg )
    {}

    /// Constructs BLAS error with message: "msg, in function <func>"
    Error( const char* msg, const char* func ):
        std::exception(),
        msg_( std::string(msg) + ", in function " + func )
    {}

    /// Returns BLAS error message
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }

private:
    std::string msg_;
};

// -----------------------------------------------------------------------------
/// Main function to handle errors in <T>BLAS
/// Default implementation: throw blas::Error( error_msg, func )
void error( const char* error_msg, const char* func );

// -----------------------------------------------------------------------------
// Internal helpers
namespace internal {

    // -------------------------------------------------------------------------
    /// internal helper function that calls blas::error if cond is true
    /// called by blas_error_if macro
    inline void error_if( bool cond, const char* condstr, const char* func )
    {
        if (cond)
            error( condstr, func );
    }

    // -------------------------------------------------------------------------
    /// internal helper function that calls blas::error if cond is true
    /// uses printf-style format for error message
    /// called by blas_error_if_msg macro
    /// condstr is ignored, but differentiates this from the other version.
    inline void error_if( bool cond, const char* condstr, const char* func,
        const char* format, ... )
    #ifndef _MSC_VER
        __attribute__((format( printf, 4, 5 )));
    #endif
    
    inline void error_if( bool cond, const char* condstr, const char* func,
        const char* format, ... )
    {
        if (cond) {
            char buf[80];
            va_list va;
            va_start( va, format );
            vsnprintf( buf, sizeof(buf), format, va );

            error( buf, func );
        }
    }

}  // namespace internal

// -----------------------------------------------------------------------------
// Macros to handle error checks
#if defined(BLAS_ERROR_NDEBUG) || defined(NDEBUG)

    // <T>BLAS does no error checking;
    // lower level BLAS may still handle errors via xerbla
    #define blas_error( msg ) \
        ((void)0)
    #define blas_error_if( cond ) \
        ((void)0)
    #define blas_error_if_msg( cond, ... ) \
        ((void)0)

#else

    /// internal macro to the get string __func__
    /// ex: blas_error( "a < b" );
    #define blas_error( msg ) \
        blas::error( msg, __func__ )

    /// internal macro to get strings: #cond and __func__
    /// ex: blas_error_if( a < b );
    #define blas_error_if( cond ) \
        blas::internal::error_if( cond, #cond, __func__ )

    /// internal macro takes cond and printf-style format for error message.
    /// ex: blas_error_if_msg( a < b, "a %d < b %d", a, b );
    #define blas_error_if_msg( cond, ... ) \
        blas::internal::error_if( cond, #cond, __func__, __VA_ARGS__ )

#endif

} // namespace blas

#endif // __TBLAS_UTILS_HH__
