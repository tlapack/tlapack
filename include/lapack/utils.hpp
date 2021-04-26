#ifndef __UTILS_HH__
#define __UTILS_HH__

#include <math.h>
#include <complex>
#include <limits>

namespace lapack {

// -----------------------------------------------------------------------------
// Based on C++14 common_type implementation from
// http://www.cplusplus.com/reference/type_traits/common_type/
// Adds promotion of complex types based on the common type of the associated
// real types. This fixes various cases:
//
// std::common_type_t< double, complex<float> > is complex<float>  (wrong)
//        scalar_type< double, complex<float> > is complex<double> (right)
//
// std::common_type_t< int, complex<long> > is not defined (compile error)
//        scalar_type< int, complex<long> > is complex<long> (right)

// for zero types
template< typename... Types >
struct scalar_type_traits;

// define scalar_type<> type alias
template< typename... Types >
using scalar_type = typename scalar_type_traits< Types... >::type;

// for one type
template< typename T >
struct scalar_type_traits< T >
{
    using type = decay_t<T>;
};

// for two types
// relies on type of ?: operator being the common type of its two arguments
template< typename T1, typename T2 >
struct scalar_type_traits< T1, T2 >
{
    using type = decay_t< decltype( true ? std::declval<T1>() : std::declval<T2>() ) >;
};

// for either or both complex,
// find common type of associated real types, then add complex
template< typename T1, typename T2 >
struct scalar_type_traits< std::complex<T1>, T2 >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

template< typename T1, typename T2 >
struct scalar_type_traits< T1, std::complex<T2> >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

template< typename T1, typename T2 >
struct scalar_type_traits< std::complex<T1>, std::complex<T2> >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

// for three or more types
template< typename T1, typename T2, typename... Types >
struct scalar_type_traits< T1, T2, Types... >
{
    using type = scalar_type< scalar_type< T1, T2 >, Types... >;
};

// -----------------------------------------------------------------------------
// for any combination of types, determine associated real, scalar,
// and complex types.
//
// real_type< float >                               is float
// real_type< float, double, complex<float> >       is double
//
// scalar_type< float >                             is float
// scalar_type< float, complex<float> >             is complex<float>
// scalar_type< float, double, complex<float> >     is complex<double>
//
// complex_type< float >                            is complex<float>
// complex_type< float, double >                    is complex<double>
// complex_type< float, double, complex<float> >    is complex<double>

// for zero types
template< typename... Types >
struct real_type_traits;

// define real_type<> type alias
template< typename... Types >
using real_type = typename real_type_traits< Types... >::real_t;

// define complex_type<> type alias
template< typename... Types >
using complex_type = std::complex< real_type< Types... > >;

// for one type
template< typename T >
struct real_type_traits<T>
{
    using real_t = T;
};

// for one complex type, strip complex
template< typename T >
struct real_type_traits< std::complex<T> >
{
    using real_t = T;
};

// for two or more types
template< typename T1, typename... Types >
struct real_type_traits< T1, Types... >
{
    using real_t = scalar_type< real_type<T1>, real_type< Types... > >;
};

// -----------------------------------------------------------------------------
// max that works with different data types
// and any number of arguments: max( a, b, c, d )

// one argument
template< typename T >
T max( T x )
{
    return x;
}

// two arguments
template< typename T1, typename T2 >
scalar_type< T1, T2 >
    max( T1 x, T2 y )
{
    return (x >= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
scalar_type< T1, Types... >
    max( T1 first, Types... args )
{
    return max( first, max( args... ) );
}

// -----------------------------------------------------------------------------
// min that works with different data types
// and any number of arguments: min( a, b, c, d )

// one argument
template< typename T >
T min( T x )
{
    return x;
}

// two arguments
template< typename T1, typename T2 >
scalar_type< T1, T2 >
    min( T1 x, T2 y )
{
    return (x <= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
scalar_type< T1, Types... >
    min( T1 first, Types... args )
{
    return min( first, min( args... ) );
}

// -----------------------------------------------------------------------------
// Generate a scalar from real and imaginary parts.
// For real scalars, the imaginary part is ignored.

// For real scalar types.
template <typename real_t>
struct MakeScalarTraits {
    static real_t make( real_t re, real_t im )
        { return re; }
};

// For complex scalar types.
template <typename real_t>
struct MakeScalarTraits< std::complex<real_t> > {
    static std::complex<real_t> make( real_t re, real_t im )
        { return std::complex<real_t>( re, im ); }
};

template <typename scalar_t>
scalar_t make_scalar( real_type<scalar_t> re,
                      real_type<scalar_t> im=0 )
{
    return MakeScalarTraits<scalar_t>::make( re, im );
}

// -----------------------------------------------------------------------------
// Macros to compute scaling constants
//
// __Further details__
//
// Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
// ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665

// Unit in Last Place
template <typename real_t>
inline const real_t ulp()
{
    return std::numeric_limits< real_t >::epsilon();
}

// Safe Minimum such that 1/safe_min() is representable
template <typename real_t>
inline const real_t safe_min()
{
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return max( pow(fradix, expm-1), pow(fradix, 1-expM) );
}

// Safe Maximum such that 1/safe_max() is representable (SAFMAX := 1/SAFMIN)
template <typename real_t>
inline const real_t safe_max()
{
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return min( pow(fradix, 1-expm), pow(fradix, expM-1) );
}

// Safe Minimum such its square is representable
template <typename real_t>
inline const real_t root_min()
{
    return sqrt( safe_min<real_t>() / ulp<real_t>() );
}

// Safe Maximum such that its square is representable
template <typename real_t>
inline const real_t root_max()
{
    return sqrt( safe_max<real_t>() * ulp<real_t>() );
}

} // namespace lapack

#endif // __UTILS_HH__