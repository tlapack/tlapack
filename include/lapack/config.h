#ifndef __TLAPACK_CONFIG_H__
#define __TLAPACK_CONFIG_H__

#if defined(BLAS_ILP64) && ! defined(LAPACK_ILP64)
    #define LAPACK_ILP64
#endif

#ifndef lapack_int
    #ifdef LAPACK_ILP64
        #define lapack_int int64_t
    #else
        #define lapack_int int
    #endif
#endif

#ifndef lapack_logical
    #define lapack_logical lapack_int
#endif

// =============================================================================
// Callback logical functions of one, two, or three arguments are used
// to select eigenvalues to sort to the top left of the Schur form in gees and gges.
// The value is selected if function returns TRUE (non-zero).

typedef lapack_logical (*lapack_s_select2) ( float const* omega_real, float const* omega_imag );
typedef lapack_logical (*lapack_s_select3) ( float const* alpha_real, float const* alpha_imag, float const* beta );

typedef lapack_logical (*lapack_d_select2) ( double const* omega_real, double const* omega_imag );
typedef lapack_logical (*lapack_d_select3) ( double const* alpha_real, double const* alpha_imag, double const* beta );

typedef lapack_logical (*lapack_c_select1) ( std::complex<float> const* omega );
typedef lapack_logical (*lapack_c_select2) ( std::complex<float> const* alpha, std::complex<float> const* beta );

typedef lapack_logical (*lapack_z_select1) ( std::complex<double> const* omega );
typedef lapack_logical (*lapack_z_select2) ( std::complex<double> const* alpha, std::complex<double> const* beta );

#endif // __TLAPACK_CONFIG_H__