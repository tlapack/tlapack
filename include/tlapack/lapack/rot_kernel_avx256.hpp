#ifndef rot_kernel_avx256_hpp
#define rot_kernel_avx256_hpp

#include <immintrin.h>

#include <complex>

//
// AVX implementations for rot_nofuse
//

/**
 * @copydoc rot_nofuse
 *
 * @note This is a specialization of rot_nofuse for AVX2.
 */
template <typename idx_t>
void rot_nofuse(
    const idx_t n, float* x1, float* x2, const float c, const float s)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256 c_v = _mm256_broadcast_ss(&c);
    __m256 s_v = _mm256_broadcast_ss(&s);

    float* x1_ptr = x1;
    float* x2_ptr = x2;

    idx_t i = 0;
    for (; i + 7 < n; i += 8) {
        // Load the vectors x1, x2
        __m256 x1_v = _mm256_loadu_ps(x1_ptr);
        __m256 x2_v = _mm256_loadu_ps(x2_ptr);

        __m256 temp = c_v * x1_v + s_v * x2_v;
        x2_v = -s_v * x1_v + c_v * x2_v;
        x1_v = temp;

        _mm256_storeu_ps(x1_ptr, x1_v);
        _mm256_storeu_ps(x2_ptr, x2_v);

        x1_ptr += 8;
        x2_ptr += 8;
    }

    // In case n is not a multiple of 8, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        float temp = c * x1[i] + s * x2[i];
        x2[i] = -s * x1[i] + c * x2[i];
        x1[i] = temp;
    }
}

/**
 * @copydoc rot_nofuse
 *
 * @note This is a specialization of rot_nofuse for AVX2.
 */
template <typename idx_t>
void rot_nofuse(
    const idx_t n, double* x1, double* x2, const double c, const double s)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256d c_v = _mm256_broadcast_sd(&c);
    __m256d s_v = _mm256_broadcast_sd(&s);

    double* x1_ptr = x1;
    double* x2_ptr = x2;

    idx_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load the vectors x1, x2
        __m256d x1_v = _mm256_loadu_pd(x1_ptr);
        __m256d x2_v = _mm256_loadu_pd(x2_ptr);

        __m256d temp = c_v * x1_v + s_v * x2_v;
        x2_v = -s_v * x1_v + c_v * x2_v;
        x1_v = temp;

        _mm256_storeu_pd(x1_ptr, x1_v);
        _mm256_storeu_pd(x2_ptr, x2_v);

        x1_ptr += 4;
        x2_ptr += 4;
    }

    // In case n is not a multiple of 4, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        double temp = c * x1[i] + s * x2[i];
        x2[i] = -s * x1[i] + c * x2[i];
        x1[i] = temp;
    }
}

/**
 * @copydoc rot_nofuse
 *
 * @note This is a specialization of rot_nofuse for AVX2.
 */
template <typename idx_t>
void rot_nofuse(const idx_t n,
                std::complex<float>* x1,
                std::complex<float>* x2,
                const float c,
                const float s)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256 c_v = _mm256_broadcast_ss(&c);
    __m256 s_v = _mm256_broadcast_ss(&s);

    // Note: the cast here should be safe, as the memory layout of
    // std::complex<float> is guaranteed to be the same as two consecutive
    // floats.
    float* x1_ptr = reinterpret_cast<float*>(x1);
    float* x2_ptr = reinterpret_cast<float*>(x2);

    idx_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load the vectors x1, x2
        __m256 x1_v = _mm256_loadu_ps(x1_ptr);
        __m256 x2_v = _mm256_loadu_ps(x2_ptr);

        __m256 temp = c_v * x1_v + s_v * x2_v;
        x2_v = -s_v * x1_v + c_v * x2_v;
        x1_v = temp;

        _mm256_storeu_ps(x1_ptr, x1_v);
        _mm256_storeu_ps(x2_ptr, x2_v);

        // Increment with 8, even though we are working with complex numbers
        // because 4 complex = 8 floats
        x1_ptr += 8;
        x2_ptr += 8;
    }

    // In case n is not a multiple of 8, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        std::complex<float> temp = c * x1[i] + s * x2[i];
        x2[i] = -s * x1[i] + c * x2[i];
        x1[i] = temp;
    }
}

/**
 * @copydoc rot_nofuse
 *
 * @note This is a specialization of rot_nofuse for AVX2.
 */
template <typename idx_t>
void rot_nofuse(const idx_t n,
                std::complex<double>* x1,
                std::complex<double>* x2,
                const double c,
                const double s)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256d c_v = _mm256_broadcast_sd(&c);
    __m256d s_v = _mm256_broadcast_sd(&s);

    // Note: the cast here should be safe, as the memory layout of
    // std::complex<double> is guaranteed to be the same as two consecutive
    // double.
    double* x1_ptr = reinterpret_cast<double*>(x1);
    double* x2_ptr = reinterpret_cast<double*>(x2);

    idx_t i = 0;
    for (; i + 1 < n; i += 2) {
        // Load the vectors x1, x2
        __m256d x1_v = _mm256_loadu_pd(x1_ptr);
        __m256d x2_v = _mm256_loadu_pd(x2_ptr);

        __m256d temp = c_v * x1_v + s_v * x2_v;
        x2_v = -s_v * x1_v + c_v * x2_v;
        x1_v = temp;

        _mm256_storeu_pd(x1_ptr, x1_v);
        _mm256_storeu_pd(x2_ptr, x2_v);

        // Increment with 4, even though we are working with complex numbers
        // because 2 complex = 4 doubles
        x1_ptr += 4;
        x2_ptr += 4;
    }

    // In case n is not a multiple of 8, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        std::complex<double> temp = c * x1[i] + s * x2[i];
        x2[i] = -s * x1[i] + c * x2[i];
        x1[i] = temp;
    }
}

//
// AVX implementations for rot_fuse2x1
//

/**
 * @copydoc rot_fuse2x1
 *
 * @note This is a specialization of rot_fuse2x1 for AVX2.
 */
template <typename idx_t>
void rot_fuse2x1(const idx_t n,
                 double* x1,
                 double* x2,
                 double* x3,
                 const double c1,
                 const double s1,
                 const double c2,
                 const double s2)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256d c1_v = _mm256_broadcast_sd(&c1);
    __m256d s1_v = _mm256_broadcast_sd(&s1);
    __m256d c2_v = _mm256_broadcast_sd(&c2);
    __m256d s2_v = _mm256_broadcast_sd(&s2);

    double* x1_ptr = &x1[0];
    double* x2_ptr = &x2[0];
    double* x3_ptr = &x3[0];

    idx_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load the vectors x1, x2, x3
        __m256d x1_v = _mm256_loadu_pd(x1_ptr);
        __m256d x2_v = _mm256_loadu_pd(x2_ptr);
        __m256d x3_v = _mm256_loadu_pd(x3_ptr);

        __m256d x2_g1 = -s1_v * x1_v + c1_v * x2_v;
        x1_v = c1_v * x1_v + s1_v * x2_v;
        x2_v = c2_v * x2_g1 + s2_v * x3_v;
        x3_v = -s2_v * x2_g1 + c2_v * x3_v;

        _mm256_storeu_pd(x1_ptr, x1_v);
        _mm256_storeu_pd(x2_ptr, x2_v);
        _mm256_storeu_pd(x3_ptr, x3_v);

        x1_ptr += 4;
        x2_ptr += 4;
        x3_ptr += 4;
    }

    // In case n is not a multiple of 4, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        double x2_g1 = -s1 * x1[i] + c1 * x2[i];
        x1[i] = c1 * x1[i] + s1 * x2[i];
        x2[i] = c2 * x2_g1 + s2 * x3[i];
        x3[i] = -s2 * x2_g1 + c2 * x3[i];
    }
}

/**
 * @copydoc rot_fuse2x1
 *
 * @note This is a specialization of rot_fuse2x1 for AVX2.
 */
template <typename idx_t>
void rot_fuse2x1(const idx_t n,
                 float* x1,
                 float* x2,
                 float* x3,
                 const float c1,
                 const float s1,
                 const float c2,
                 const float s2)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256 c1_v = _mm256_broadcast_ss(&c1);
    __m256 s1_v = _mm256_broadcast_ss(&s1);
    __m256 c2_v = _mm256_broadcast_ss(&c2);
    __m256 s2_v = _mm256_broadcast_ss(&s2);

    float* x1_ptr = &x1[0];
    float* x2_ptr = &x2[0];
    float* x3_ptr = &x3[0];

    idx_t i = 0;
    for (; i + 7 < n; i += 8) {
        // Load the vectors x1, x2, x3
        __m256 x1_v = _mm256_loadu_ps(x1_ptr);
        __m256 x2_v = _mm256_loadu_ps(x2_ptr);
        __m256 x3_v = _mm256_loadu_ps(x3_ptr);

        __m256 x2_g1 = -s1_v * x1_v + c1_v * x2_v;
        x1_v = c1_v * x1_v + s1_v * x2_v;
        x2_v = c2_v * x2_g1 + s2_v * x3_v;
        x3_v = -s2_v * x2_g1 + c2_v * x3_v;

        _mm256_storeu_ps(x1_ptr, x1_v);
        _mm256_storeu_ps(x2_ptr, x2_v);
        _mm256_storeu_ps(x3_ptr, x3_v);

        x1_ptr += 8;
        x2_ptr += 8;
        x3_ptr += 8;
    }

    // In case n is not a multiple of 8, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        float x2_g1 = -s1 * x1[i] + c1 * x2[i];
        x1[i] = c1 * x1[i] + s1 * x2[i];
        x2[i] = c2 * x2_g1 + s2 * x3[i];
        x3[i] = -s2 * x2_g1 + c2 * x3[i];
    }
}

/**
 * @copydoc rot_fuse2x1
 *
 * @note This is a specialization of rot_fuse2x1 for AVX2.
 */
template <typename idx_t>
void rot_fuse2x1(const idx_t n,
                 std::complex<double>* x1,
                 std::complex<double>* x2,
                 std::complex<double>* x3,
                 const double c1,
                 const double s1,
                 const double c2,
                 const double s2)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256d c1_v = _mm256_broadcast_sd(&c1);
    __m256d s1_v = _mm256_broadcast_sd(&s1);
    __m256d c2_v = _mm256_broadcast_sd(&c2);
    __m256d s2_v = _mm256_broadcast_sd(&s2);

    double* x1_ptr = reinterpret_cast<double*>(x1);
    double* x2_ptr = reinterpret_cast<double*>(x2);
    double* x3_ptr = reinterpret_cast<double*>(x3);

    idx_t i = 0;
    for (; i + 1 < n; i += 2) {
        // Load the vectors x1, x2, x3
        __m256d x1_v = _mm256_loadu_pd(x1_ptr);
        __m256d x2_v = _mm256_loadu_pd(x2_ptr);
        __m256d x3_v = _mm256_loadu_pd(x3_ptr);

        __m256d x2_g1 = -s1_v * x1_v + c1_v * x2_v;
        x1_v = c1_v * x1_v + s1_v * x2_v;
        x2_v = c2_v * x2_g1 + s2_v * x3_v;
        x3_v = -s2_v * x2_g1 + c2_v * x3_v;

        _mm256_storeu_pd(x1_ptr, x1_v);
        _mm256_storeu_pd(x2_ptr, x2_v);
        _mm256_storeu_pd(x3_ptr, x3_v);

        x1_ptr += 4;
        x2_ptr += 4;
        x3_ptr += 4;
    }

    // In case n is not a multiple of 2, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        std::complex<double> x2_g1 = -s1 * x1[i] + c1 * x2[i];
        x1[i] = c1 * x1[i] + s1 * x2[i];
        x2[i] = c2 * x2_g1 + s2 * x3[i];
        x3[i] = -s2 * x2_g1 + c2 * x3[i];
    }
}

/**
 * @copydoc rot_fuse2x1
 *
 * @note This is a specialization of rot_fuse2x1 for AVX2.
 */
template <typename idx_t>
void rot_fuse2x1(const idx_t n,
                 std::complex<float>* x1,
                 std::complex<float>* x2,
                 std::complex<float>* x3,
                 const float c1,
                 const float s1,
                 const float c2,
                 const float s2)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256 c1_v = _mm256_broadcast_ss(&c1);
    __m256 s1_v = _mm256_broadcast_ss(&s1);
    __m256 c2_v = _mm256_broadcast_ss(&c2);
    __m256 s2_v = _mm256_broadcast_ss(&s2);

    float* x1_ptr = reinterpret_cast<float*>(x1);
    float* x2_ptr = reinterpret_cast<float*>(x2);
    float* x3_ptr = reinterpret_cast<float*>(x3);

    idx_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load the vectors x1, x2, x3
        __m256 x1_v = _mm256_loadu_ps(x1_ptr);
        __m256 x2_v = _mm256_loadu_ps(x2_ptr);
        __m256 x3_v = _mm256_loadu_ps(x3_ptr);

        __m256 x2_g1 = -s1_v * x1_v + c1_v * x2_v;
        x1_v = c1_v * x1_v + s1_v * x2_v;
        x2_v = c2_v * x2_g1 + s2_v * x3_v;
        x3_v = -s2_v * x2_g1 + c2_v * x3_v;

        _mm256_storeu_ps(x1_ptr, x1_v);
        _mm256_storeu_ps(x2_ptr, x2_v);
        _mm256_storeu_ps(x3_ptr, x3_v);

        x1_ptr += 8;
        x2_ptr += 8;
        x3_ptr += 8;
    }

    // In case n is not a multiple of 8, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        std::complex<float> x2_g1 = -s1 * x1[i] + c1 * x2[i];
        x1[i] = c1 * x1[i] + s1 * x2[i];
        x2[i] = c2 * x2_g1 + s2 * x3[i];
        x3[i] = -s2 * x2_g1 + c2 * x3[i];
    }
}

//
// AVX implementations for rot_fuse1x2
//

/**
 * @copydoc rot_fuse1x2
 *
 * @note This is a specialization of rot_fuse1x2 for AVX2.
 */
template <typename idx_t>
void rot_fuse1x2(const idx_t n,
                 double* x1,
                 double* x2,
                 double* x3,
                 const double c1,
                 const double s1,
                 const double c2,
                 const double s2)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256d c1_v = _mm256_broadcast_sd(&c1);
    __m256d s1_v = _mm256_broadcast_sd(&s1);
    __m256d c2_v = _mm256_broadcast_sd(&c2);
    __m256d s2_v = _mm256_broadcast_sd(&s2);

    double* x1_ptr = &x1[0];
    double* x2_ptr = &x2[0];
    double* x3_ptr = &x3[0];

    idx_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load the vectors x1, x2, x3
        __m256d x1_v = _mm256_loadu_pd(x1_ptr);
        __m256d x2_v = _mm256_loadu_pd(x2_ptr);
        __m256d x3_v = _mm256_loadu_pd(x3_ptr);

        __m256d x2_g1 = c1_v * x2_v + s1_v * x3_v;
        x3_v = -s1_v * x2_v + c1_v * x3_v;
        x2_v = -s2_v * x1_v + c2_v * x2_g1;
        x1_v = c2_v * x1_v + s2_v * x2_g1;

        _mm256_storeu_pd(x1_ptr, x1_v);
        _mm256_storeu_pd(x2_ptr, x2_v);
        _mm256_storeu_pd(x3_ptr, x3_v);

        x1_ptr += 4;
        x2_ptr += 4;
        x3_ptr += 4;
    }

    // In case n is not a multiple of 4, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        double x2_g1 = c1 * x2[i] + s1 * x3[i];
        x3[i] = -s1 * x2[i] + c1 * x3[i];
        x2[i] = -s2 * x1[i] + c2 * x2_g1;
        x1[i] = c2 * x1[i] + s2 * x2_g1;
    }
}

/**
 * @copydoc rot_fuse1x2
 *
 * @note This is a specialization of rot_fuse1x2 for AVX2.
 */
template <typename idx_t>
void rot_fuse1x2(const idx_t n,
                 float* x1,
                 float* x2,
                 float* x3,
                 const float c1,
                 const float s1,
                 const float c2,
                 const float s2)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256 c1_v = _mm256_broadcast_ss(&c1);
    __m256 s1_v = _mm256_broadcast_ss(&s1);
    __m256 c2_v = _mm256_broadcast_ss(&c2);
    __m256 s2_v = _mm256_broadcast_ss(&s2);

    float* x1_ptr = &x1[0];
    float* x2_ptr = &x2[0];
    float* x3_ptr = &x3[0];

    idx_t i = 0;
    for (; i + 7 < n; i += 8) {
        // Load the vectors x1, x2, x3
        __m256 x1_v = _mm256_loadu_ps(x1_ptr);
        __m256 x2_v = _mm256_loadu_ps(x2_ptr);
        __m256 x3_v = _mm256_loadu_ps(x3_ptr);

        __m256 x2_g1 = c1_v * x2_v + s1_v * x3_v;
        x3_v = -s1_v * x2_v + c1_v * x3_v;
        x2_v = -s2_v * x1_v + c2_v * x2_g1;
        x1_v = c2_v * x1_v + s2_v * x2_g1;

        _mm256_storeu_ps(x1_ptr, x1_v);
        _mm256_storeu_ps(x2_ptr, x2_v);
        _mm256_storeu_ps(x3_ptr, x3_v);

        x1_ptr += 8;
        x2_ptr += 8;
        x3_ptr += 8;
    }

    // In case n is not a multiple of 8, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        float x2_g1 = c1 * x2[i] + s1 * x3[i];
        x3[i] = -s1 * x2[i] + c1 * x3[i];
        x2[i] = -s2 * x1[i] + c2 * x2_g1;
        x1[i] = c2 * x1[i] + s2 * x2_g1;
    }
}

/**
 * @copydoc rot_fuse1x2
 *
 * @note This is a specialization of rot_fuse1x2 for AVX2.
 */
template <typename idx_t>
void rot_fuse1x2(const idx_t n,
                 std::complex<double>* x1,
                 std::complex<double>* x2,
                 std::complex<double>* x3,
                 const double c1,
                 const double s1,
                 const double c2,
                 const double s2)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256d c1_v = _mm256_broadcast_sd(&c1);
    __m256d s1_v = _mm256_broadcast_sd(&s1);
    __m256d c2_v = _mm256_broadcast_sd(&c2);
    __m256d s2_v = _mm256_broadcast_sd(&s2);

    double* x1_ptr = reinterpret_cast<double*>(x1);
    double* x2_ptr = reinterpret_cast<double*>(x2);
    double* x3_ptr = reinterpret_cast<double*>(x3);

    idx_t i = 0;
    for (; i + 1 < n; i += 2) {
        // Load the vectors x1, x2, x3
        __m256d x1_v = _mm256_loadu_pd(x1_ptr);
        __m256d x2_v = _mm256_loadu_pd(x2_ptr);
        __m256d x3_v = _mm256_loadu_pd(x3_ptr);

        __m256d x2_g1 = c1_v * x2_v + s1_v * x3_v;
        x3_v = -s1_v * x2_v + c1_v * x3_v;
        x2_v = -s2_v * x1_v + c2_v * x2_g1;
        x1_v = c2_v * x1_v + s2_v * x2_g1;

        _mm256_storeu_pd(x1_ptr, x1_v);
        _mm256_storeu_pd(x2_ptr, x2_v);
        _mm256_storeu_pd(x3_ptr, x3_v);

        x1_ptr += 4;
        x2_ptr += 4;
        x3_ptr += 4;
    }

    // In case n is not a multiple of 2, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        std::complex<double> x2_g1 = c1 * x2[i] + s1 * x3[i];
        x3[i] = -s1 * x2[i] + c1 * x3[i];
        x2[i] = -s2 * x1[i] + c2 * x2_g1;
        x1[i] = c2 * x1[i] + s2 * x2_g1;
    }
}

/**
 * @copydoc rot_fuse1x2
 *
 * @note This is a specialization of rot_fuse1x2 for AVX2.
 */
template <typename idx_t>
void rot_fuse1x2(const idx_t n,
                 std::complex<float>* x1,
                 std::complex<float>* x2,
                 std::complex<float>* x3,
                 const float c1,
                 const float s1,
                 const float c2,
                 const float s2)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256 c1_v = _mm256_broadcast_ss(&c1);
    __m256 s1_v = _mm256_broadcast_ss(&s1);
    __m256 c2_v = _mm256_broadcast_ss(&c2);
    __m256 s2_v = _mm256_broadcast_ss(&s2);

    float* x1_ptr = reinterpret_cast<float*>(x1);
    float* x2_ptr = reinterpret_cast<float*>(x2);
    float* x3_ptr = reinterpret_cast<float*>(x3);

    idx_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load the vectors x1, x2, x3
        __m256 x1_v = _mm256_loadu_ps(x1_ptr);
        __m256 x2_v = _mm256_loadu_ps(x2_ptr);
        __m256 x3_v = _mm256_loadu_ps(x3_ptr);

        __m256 x2_g1 = c1_v * x2_v + s1_v * x3_v;
        x3_v = -s1_v * x2_v + c1_v * x3_v;
        x2_v = -s2_v * x1_v + c2_v * x2_g1;
        x1_v = c2_v * x1_v + s2_v * x2_g1;
        _mm256_storeu_ps(x1_ptr, x1_v);
        _mm256_storeu_ps(x2_ptr, x2_v);
        _mm256_storeu_ps(x3_ptr, x3_v);

        x1_ptr += 8;
        x2_ptr += 8;
        x3_ptr += 8;
    }

    // In case n is not a multiple of 8, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        std::complex<float> x2_g1 = c1 * x2[i] + s1 * x3[i];
        x3[i] = -s1 * x2[i] + c1 * x3[i];
        x2[i] = -s2 * x1[i] + c2 * x2_g1;
        x1[i] = c2 * x1[i] + s2 * x2_g1;
    }
}

//
// AVX implementations for rot_fuse2x2
//

/**
 * @copydoc rot_fuse2x2
 *
 * @note This is a specialization of rot_fuse2x2 for AVX2.
 */
template <typename idx_t>
void rot_fuse2x2(const idx_t n,
                 double* x1,
                 double* x2,
                 double* x3,
                 double* x4,
                 const double c1,
                 const double s1,
                 const double c2,
                 const double s2,
                 const double c3,
                 const double s3,
                 const double c4,
                 const double s4)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256d c1_v = _mm256_broadcast_sd(&c1);
    __m256d s1_v = _mm256_broadcast_sd(&s1);
    __m256d c2_v = _mm256_broadcast_sd(&c2);
    __m256d s2_v = _mm256_broadcast_sd(&s2);
    __m256d c3_v = _mm256_broadcast_sd(&c3);
    __m256d s3_v = _mm256_broadcast_sd(&s3);
    __m256d c4_v = _mm256_broadcast_sd(&c4);
    __m256d s4_v = _mm256_broadcast_sd(&s4);

    double* x1_ptr = &x1[0];
    double* x2_ptr = &x2[0];
    double* x3_ptr = &x3[0];
    double* x4_ptr = &x4[0];

    idx_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load the vectors x1, x2, x3, x4
        __m256d x1_v = _mm256_loadu_pd(x1_ptr);
        __m256d x2_v = _mm256_loadu_pd(x2_ptr);
        __m256d x3_v = _mm256_loadu_pd(x3_ptr);
        __m256d x4_v = _mm256_loadu_pd(x4_ptr);

        // Apply the Givens rotations
        __m256d x2_g1 = c1_v * x2_v + s1_v * x3_v;
        __m256d x3_g1 = -s1_v * x2_v + c1_v * x3_v;
        __m256d x2_g2 = -s2_v * x1_v + c2_v * x2_g1;
        x1_v = c2_v * x1_v + s2_v * x2_g1;
        __m256d x3_g3 = c3_v * x3_g1 + s3_v * x4_v;
        x4_v = -s3_v * x3_g1 + c3_v * x4_v;
        x2_v = c4_v * x2_g2 + s4_v * x3_g3;
        x3_v = -s4_v * x2_g2 + c4_v * x3_g3;

        _mm256_storeu_pd(x1_ptr, x1_v);
        _mm256_storeu_pd(x2_ptr, x2_v);
        _mm256_storeu_pd(x3_ptr, x3_v);
        _mm256_storeu_pd(x4_ptr, x4_v);

        x1_ptr += 4;
        x2_ptr += 4;
        x3_ptr += 4;
        x4_ptr += 4;
    }

    // In case n is not a multiple of 4, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        double x2_g1 = c1 * x2[i] + s1 * x3[i];
        double x3_g1 = -s1 * x2[i] + c1 * x3[i];
        double x2_g2 = -s2 * x1[i] + c2 * x2_g1;
        x1[i] = c2 * x1[i] + s2 * x2_g1;
        double x3_g3 = c3 * x3_g1 + s3 * x4[i];
        x4[i] = -s3 * x3_g1 + c3 * x4[i];
        x2[i] = c4 * x2_g2 + s4 * x3_g3;
        x3[i] = -s4 * x2_g2 + c4 * x3_g3;
    }
}

/**
 * @copydoc rot_fuse2x2
 *
 * @note This is a specialization of rot_fuse2x2 for AVX2.
 */
template <typename idx_t>
void rot_fuse2x2(const idx_t n,
                 float* x1,
                 float* x2,
                 float* x3,
                 float* x4,
                 const float c1,
                 const float s1,
                 const float c2,
                 const float s2,
                 const float c3,
                 const float s3,
                 const float c4,
                 const float s4)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256 c1_v = _mm256_broadcast_ss(&c1);
    __m256 s1_v = _mm256_broadcast_ss(&s1);
    __m256 c2_v = _mm256_broadcast_ss(&c2);
    __m256 s2_v = _mm256_broadcast_ss(&s2);
    __m256 c3_v = _mm256_broadcast_ss(&c3);
    __m256 s3_v = _mm256_broadcast_ss(&s3);
    __m256 c4_v = _mm256_broadcast_ss(&c4);
    __m256 s4_v = _mm256_broadcast_ss(&s4);

    float* x1_ptr = &x1[0];
    float* x2_ptr = &x2[0];
    float* x3_ptr = &x3[0];
    float* x4_ptr = &x4[0];

    idx_t i = 0;
    for (; i + 7 < n; i += 8) {
        // Load the vectors x1, x2, x3, x4
        __m256 x1_v = _mm256_loadu_ps(x1_ptr);
        __m256 x2_v = _mm256_loadu_ps(x2_ptr);
        __m256 x3_v = _mm256_loadu_ps(x3_ptr);
        __m256 x4_v = _mm256_loadu_ps(x4_ptr);

        // Apply the Givens rotations
        __m256 x2_g1 = c1_v * x2_v + s1_v * x3_v;
        __m256 x3_g1 = -s1_v * x2_v + c1_v * x3_v;
        __m256 x2_g2 = -s2_v * x1_v + c2_v * x2_g1;
        x1_v = c2_v * x1_v + s2_v * x2_g1;
        __m256 x3_g3 = c3_v * x3_g1 + s3_v * x4_v;
        x4_v = -s3_v * x3_g1 + c3_v * x4_v;
        x2_v = c4_v * x2_g2 + s4_v * x3_g3;
        x3_v = -s4_v * x2_g2 + c4_v * x3_g3;

        _mm256_storeu_ps(x1_ptr, x1_v);
        _mm256_storeu_ps(x2_ptr, x2_v);
        _mm256_storeu_ps(x3_ptr, x3_v);
        _mm256_storeu_ps(x4_ptr, x4_v);

        x1_ptr += 8;
        x2_ptr += 8;
        x3_ptr += 8;
        x4_ptr += 8;
    }

    // In case n is not a multiple of 4, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        float x2_g1 = c1 * x2[i] + s1 * x3[i];
        float x3_g1 = -s1 * x2[i] + c1 * x3[i];
        float x2_g2 = -s2 * x1[i] + c2 * x2_g1;
        x1[i] = c2 * x1[i] + s2 * x2_g1;
        float x3_g3 = c3 * x3_g1 + s3 * x4[i];
        x4[i] = -s3 * x3_g1 + c3 * x4[i];
        x2[i] = c4 * x2_g2 + s4 * x3_g3;
        x3[i] = -s4 * x2_g2 + c4 * x3_g3;
    }
}

/**
 * @copydoc rot_fuse2x2
 *
 * @note This is a specialization of rot_fuse2x2 for AVX2.
 */
template <typename idx_t>
void rot_fuse2x2(const idx_t n,
                 std::complex<double>* x1,
                 std::complex<double>* x2,
                 std::complex<double>* x3,
                 std::complex<double>* x4,
                 const double c1,
                 const double s1,
                 const double c2,
                 const double s2,
                 const double c3,
                 const double s3,
                 const double c4,
                 const double s4)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256d c1_v = _mm256_broadcast_sd(&c1);
    __m256d s1_v = _mm256_broadcast_sd(&s1);
    __m256d c2_v = _mm256_broadcast_sd(&c2);
    __m256d s2_v = _mm256_broadcast_sd(&s2);
    __m256d c3_v = _mm256_broadcast_sd(&c3);
    __m256d s3_v = _mm256_broadcast_sd(&s3);
    __m256d c4_v = _mm256_broadcast_sd(&c4);
    __m256d s4_v = _mm256_broadcast_sd(&s4);

    double* x1_ptr = reinterpret_cast<double*>(x1);
    double* x2_ptr = reinterpret_cast<double*>(x2);
    double* x3_ptr = reinterpret_cast<double*>(x3);
    double* x4_ptr = reinterpret_cast<double*>(x4);

    idx_t i = 0;
    for (; i + 1 < n; i += 2) {
        // Load the vectors x1, x2, x3, x4
        __m256d x1_v = _mm256_loadu_pd(x1_ptr);
        __m256d x2_v = _mm256_loadu_pd(x2_ptr);
        __m256d x3_v = _mm256_loadu_pd(x3_ptr);
        __m256d x4_v = _mm256_loadu_pd(x4_ptr);

        // Apply the Givens rotations
        __m256d x2_g1 = c1_v * x2_v + s1_v * x3_v;
        __m256d x3_g1 = -s1_v * x2_v + c1_v * x3_v;
        __m256d x2_g2 = -s2_v * x1_v + c2_v * x2_g1;
        x1_v = c2_v * x1_v + s2_v * x2_g1;
        __m256d x3_g3 = c3_v * x3_g1 + s3_v * x4_v;
        x4_v = -s3_v * x3_g1 + c3_v * x4_v;
        x2_v = c4_v * x2_g2 + s4_v * x3_g3;
        x3_v = -s4_v * x2_g2 + c4_v * x3_g3;

        _mm256_storeu_pd(x1_ptr, x1_v);
        _mm256_storeu_pd(x2_ptr, x2_v);
        _mm256_storeu_pd(x3_ptr, x3_v);
        _mm256_storeu_pd(x4_ptr, x4_v);

        x1_ptr += 4;
        x2_ptr += 4;
        x3_ptr += 4;
        x4_ptr += 4;
    }

    // In case n is not a multiple of 2, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        std::complex<double> x2_g1 = c1 * x2[i] + s1 * x3[i];
        std::complex<double> x3_g1 = -s1 * x2[i] + c1 * x3[i];
        std::complex<double> x2_g2 = -s2 * x1[i] + c2 * x2_g1;
        x1[i] = c2 * x1[i] + s2 * x2_g1;
        std::complex<double> x3_g3 = c3 * x3_g1 + s3 * x4[i];
        x4[i] = -s3 * x3_g1 + c3 * x4[i];
        x2[i] = c4 * x2_g2 + s4 * x3_g3;
        x3[i] = -s4 * x2_g2 + c4 * x3_g3;
    }
}

/**
 * @copydoc rot_fuse2x2
 *
 * @note This is a specialization of rot_fuse2x2 for AVX2.
 */
template <typename idx_t>
void rot_fuse2x2(const idx_t n,
                 std::complex<float>* x1,
                 std::complex<float>* x2,
                 std::complex<float>* x3,
                 std::complex<float>* x4,
                 const float c1,
                 const float s1,
                 const float c2,
                 const float s2,
                 const float c3,
                 const float s3,
                 const float c4,
                 const float s4)
{
    // Load the Givens coefficients into SIMD registers
    // the same coefficients are used for all elements of the vectors
    // so we broadcast them to all elements of the SIMD registers
    __m256 c1_v = _mm256_broadcast_ss(&c1);
    __m256 s1_v = _mm256_broadcast_ss(&s1);
    __m256 c2_v = _mm256_broadcast_ss(&c2);
    __m256 s2_v = _mm256_broadcast_ss(&s2);
    __m256 c3_v = _mm256_broadcast_ss(&c3);
    __m256 s3_v = _mm256_broadcast_ss(&s3);
    __m256 c4_v = _mm256_broadcast_ss(&c4);
    __m256 s4_v = _mm256_broadcast_ss(&s4);

    float* x1_ptr = reinterpret_cast<float*>(x1);
    float* x2_ptr = reinterpret_cast<float*>(x2);
    float* x3_ptr = reinterpret_cast<float*>(x3);
    float* x4_ptr = reinterpret_cast<float*>(x4);

    idx_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load the vectors x1, x2, x3, x4
        __m256 x1_v = _mm256_loadu_ps(x1_ptr);
        __m256 x2_v = _mm256_loadu_ps(x2_ptr);
        __m256 x3_v = _mm256_loadu_ps(x3_ptr);
        __m256 x4_v = _mm256_loadu_ps(x4_ptr);

        // Apply the Givens rotations
        __m256 x2_g1 = c1_v * x2_v + s1_v * x3_v;
        __m256 x3_g1 = -s1_v * x2_v + c1_v * x3_v;
        __m256 x2_g2 = -s2_v * x1_v + c2_v * x2_g1;
        x1_v = c2_v * x1_v + s2_v * x2_g1;
        __m256 x3_g3 = c3_v * x3_g1 + s3_v * x4_v;
        x4_v = -s3_v * x3_g1 + c3_v * x4_v;
        x2_v = c4_v * x2_g2 + s4_v * x3_g3;
        x3_v = -s4_v * x2_g2 + c4_v * x3_g3;

        _mm256_storeu_ps(x1_ptr, x1_v);
        _mm256_storeu_ps(x2_ptr, x2_v);
        _mm256_storeu_ps(x3_ptr, x3_v);
        _mm256_storeu_ps(x4_ptr, x4_v);

        x1_ptr += 8;
        x2_ptr += 8;
        x3_ptr += 8;
        x4_ptr += 8;
    }

    // In case n is not a multiple of 4, apply the remaining rotations using the
    // scalar version
    for (; i < n; i++) {
        std::complex<float> x2_g1 = c1 * x2[i] + s1 * x3[i];
        std::complex<float> x3_g1 = -s1 * x2[i] + c1 * x3[i];
        std::complex<float> x2_g2 = -s2 * x1[i] + c2 * x2_g1;
        x1[i] = c2 * x1[i] + s2 * x2_g1;
        std::complex<float> x3_g3 = c3 * x3_g1 + s3 * x4[i];
        x4[i] = -s3 * x3_g1 + c3 * x4[i];
        x2[i] = c4 * x2_g2 + s4 * x3_g3;
        x3[i] = -s4 * x2_g2 + c4 * x3_g3;
    }
}

#endif  // rot_kernel_avx256_hpp