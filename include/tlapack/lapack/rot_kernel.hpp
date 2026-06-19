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

#ifdef __AVX512F__

    #include "rot_kernel_avx512.hpp"

#else
    #ifdef __AVX2__

        #include "rot_kernel_avx256.hpp"

    #endif  // __AVX2__
#endif      // __AVX512F__

}  // namespace tlapack

#endif  // TLAPACK_ROT_KERNEL_HH
