/// @file rot_kernel.hpp Efficient kernels for applying sequences of plane
/// rotations to a matrix
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ROT_KERNEL_HH
#define TLAPACK_ROT_KERNEL_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Applies a plane rotation to two contiguous vectors.
 *
 * This routine differs from the BLAS routine `rot` in that it
 * requires contiguous memory for the vectors.
 *
 * @param[in]     n  The number of elements in the vectors.
 * @param[in,out] x1 Pointer to the first vector.
 * @param[in,out] x2 Pointer to the second vector.
 * @param[in]     c  The cosine of the rotation.
 * @param[in]     s  The sine of the rotation.
 */
template <typename idx_t, typename X_t, typename C_t, typename S_t>
void rot_nofuse(const idx_t n, X_t* x1, X_t* x2, const C_t c, const S_t s)
{
    for (idx_t i = 0; i < n; ++i) {
        X_t temp = c * x1[i] + s * x2[i];
        x2[i] = -conj(s) * x1[i] + c * x2[i];
        x1[i] = temp;
    }
}

#ifdef __AVX2__

    #include <immintrin.h>

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

#endif  // __AVX2__

/** Applies two plane rotations to three contiguous vectors.
 *
 * [x1]   [1 0   0  ]   [c1  s1 0]   [x1]
 * [x2] = [0 c2  s2 ] * [-s1 c1 0] * [x2]
 * [x3]   [0 -s2 c2 ]   [0   0  1]   [x3]
 *
 * @param[in]     n   The number of elements in the vectors.
 * @param[in,out] x1  Pointer to the first vector.
 * @param[in,out] x2  Pointer to the second vector.
 * @param[in,out] x3  Pointer to the third vector.
 * @param[in]     c1  The cosine of the first rotation.
 * @param[in]     s1  The sine of the first rotation.
 * @param[in]     c2  The cosine of the second rotation.
 * @param[in]     s2  The sine of the second rotation.
 */
template <typename X_t, typename C_t, typename S_t, typename idx_t>
void rot_fuse2x1(
    idx_t n, X_t* x1, X_t* x2, X_t* x3, C_t c1, S_t s1, C_t c2, S_t s2)
{
    for (idx_t i = 0; i < n; ++i) {
        X_t x2_g1 = -conj(s1) * x1[i] + c1 * x2[i];
        x1[i] = c1 * x1[i] + s1 * x2[i];
        x2[i] = c2 * x2_g1 + s2 * x3[i];
        x3[i] = -conj(s2) * x2_g1 + c2 * x3[i];
    }
}

#ifdef __AVX2__

    #include <immintrin.h>

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

#endif  // __AVX2__

/** Applies two plane rotations to three contiguous vectors.
 *
 * [x1]   [c2  s2 0]   [1 0   0  ]   [x1]
 * [x2] = [-s2 c2 0] * [0 c1  s1 ] * [x2]
 * [x3]   [0   0  1]   [0 -s1 c1 ]   [x3]
 *
 * @param[in]     n   The number of elements in the vectors.
 * @param[in,out] x1  Pointer to the first vector.
 * @param[in,out] x2  Pointer to the second vector.
 * @param[in,out] x3  Pointer to the third vector.
 * @param[in]     c1  The cosine of the first rotation.
 * @param[in]     s1  The sine of the first rotation.
 * @param[in]     c2  The cosine of the second rotation.
 * @param[in]     s2  The sine of the second rotation.
 */
template <typename X_t, typename C_t, typename S_t, typename idx_t>
void rot_fuse1x2(const idx_t n,
                 X_t* x1,
                 X_t* x2,
                 X_t* x3,
                 const C_t c1,
                 const S_t s1,
                 const C_t c2,
                 const S_t s2)
{
    for (idx_t i = 0; i < n; ++i) {
        X_t x2_g1 = c1 * x2[i] + s1 * x3[i];
        x3[i] = -conj(s1) * x2[i] + c1 * x3[i];
        x2[i] = -conj(s2) * x1[i] + c2 * x2_g1;
        x1[i] = c2 * x1[i] + s2 * x2_g1;
    }
}

#ifdef __AVX2__

    #include <immintrin.h>

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

#endif  // __AVX2__

/** Applies four plane rotations to four contiguous vectors.
 *
 * [x1]   [1 0   0  0]   [1 0 0   0 ]   [c2  s2 0 0]   [1 0   0  0]   [x1]
 * [x2] = [0 c4  s4 0] * [0 1 0   0 ] * [-s2 c2 0 0] * [0 c1  s1 0] * [x2]
 * [x3]   [0 -s4 c4 0]   [0 0 c3  s3]   [0   0  1 0]   [0 -s1 c1 0]   [x3]
 * [x4]   [0 0   0  1]   [0 0 -s3 c3]   [0   0  0 1]   [0 0   0  0]   [x4]
 *
 * Note: the rotations are applied in the order G1, G2, G3, G4,
 * but the order G1, G3, G2, G4 is also possible.
 *
 * @param[in]     n   The number of elements in the vectors.
 * @param[in,out] x1  Pointer to the first vector.
 * @param[in,out] x2  Pointer to the second vector.
 * @param[in,out] x3  Pointer to the third vector.
 * @param[in,out] x4  Pointer to the fourth vector.
 * @param[in]     c1  The cosine of the first rotation.
 * @param[in]     s1  The sine of the first rotation.
 * @param[in]     c2  The cosine of the second rotation.
 * @param[in]     s2  The sine of the second rotation.
 * @param[in]     c3  The cosine of the third rotation.
 * @param[in]     s3  The sine of the third rotation.
 * @param[in]     c4  The cosine of the fourth rotation.
 * @param[in]     s4  The sine of the fourth rotation.
 */
template <typename X_t, typename C_t, typename S_t, typename idx_t>
void rot_fuse2x2(const idx_t n,
                 X_t* x1,
                 X_t* x2,
                 X_t* x3,
                 X_t* x4,
                 const C_t c1,
                 const S_t s1,
                 const C_t c2,
                 const S_t s2,
                 const C_t c3,
                 const S_t s3,
                 const C_t c4,
                 const S_t s4)
{
    for (idx_t i = 0; i < n; ++i) {
        X_t x2_g1 = c1 * x2[i] + s1 * x3[i];
        X_t x3_g1 = -conj(s1) * x2[i] + c1 * x3[i];
        X_t x2_g2 = -conj(s2) * x1[i] + c2 * x2_g1;
        x1[i] = c2 * x1[i] + s2 * x2_g1;
        X_t x3_g3 = c3 * x3_g1 + s3 * x4[i];
        x4[i] = -conj(s3) * x3_g1 + c3 * x4[i];
        x2[i] = c4 * x2_g2 + s4 * x3_g3;
        x3[i] = -conj(s4) * x2_g2 + c4 * x3_g3;
    }
}

#ifdef __AVX2__

    #include <immintrin.h>

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

#endif  // __AVX2__

}  // namespace tlapack

#endif  // TLAPACK_ROT_KERNEL_HH
