/// @file move_bulge.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __MOVE_BULGE_HH__
#define __MOVE_BULGE_HH__


#include <memory>
#include <complex>

#include "legacy_api/base/utils.hpp"
#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/lahqr_shiftcolumn.hpp"

namespace tlapack
{

    /** Given a 4-by-3 matrix H and small order reflector v,
     *  move_bulge applies the delayed right update to the last
     *  row and calculates a new reflector to move the bulge
     *  down. If the bulge collapses, an attempt is made to
     *  reintroduce it using shifts s1 and s2.
     *
     * @param[in] H 4x3 matrix.
     * @param[out] v vector of size 3
     *      On entry, the delayed reflector to apply
     *      The first element of the reflector is assumed to be one, and v[0] instead stores tau.
     *      On exit, the reflector that moves the bulge down one position
     * @param[in] s1 complex valued shift
     * @param[in] s2 complex valued shift
     *
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        typename T = type_t<matrix_t>,
        typename real_t = real_type<T>>
    void move_bulge(matrix_t &H, vector_t &v, std::complex<real_t> s1, std::complex<real_t> s2)
    {

        using idx_t = size_type<matrix_t>;
        using pair = std::pair<idx_t, idx_t>;
        const T zero(0);
        const real_t eps = uroundoff<real_t>();

        // Perform delayed update of row below the bulge
        // Assumes the first two elements of the row are zero
        auto refsum = v[0] * v[2] * H(3, 2);
        H(3, 0) = -refsum;
        H(3, 1) = -refsum * conj(v[1]);
        H(3, 2) = H(3, 2) - refsum * conj(v[2]);

        // Generate reflector to move bulge down
        T tau, beta;
        v[0] = H(1, 0);
        v[1] = H(2, 0);
        v[2] = H(3, 0);
        larfg(v, tau);
        beta = v[0];
        v[0] = tau;

        // Check for bulge collapse
        if (H(3, 0) != zero or H(3, 1) != zero or H(3, 2) != zero)
        {
            // The bulge hasn't collapsed, typical case
            H(1, 0) = beta;
            H(2, 0) = zero;
            H(3, 0) = zero;
        }
        else
        {
            // The bulge has collapsed, attempt to reintroduce using
            // 2-small-subdiagonals trick
            std::unique_ptr<T[]> _vt(new T[3]);
            auto vt = legacyVector<T>(3, &_vt[0]);
            auto H2 = slice(H, pair{1, 4}, pair{1, 4});
            lahqr_shiftcolumn(H2, vt, s1, s2);
            larfg(vt, tau);
            vt[0] = tau;

            refsum = conj(vt[0]) * H(1, 0) + conj(vt[1]) * H(2, 0);
            if (abs1(H(2, 0) - refsum * vt[1]) + abs1(refsum * vt[2]) > eps * (abs1(H(0, 0)) + abs1(H(1, 1)) + abs1(H(2, 2))))
            {
                // Starting a new bulge here would create non-negligible fill. Use the old one.
                H(1, 0) = beta;
                H(2, 0) = zero;
                H(3, 0) = zero;
            }
            else
            {
                // Fill-in is negligible, use the new reflector.
                H(1, 0) = H(1, 0) - refsum;
                H(2, 0) = zero;
                H(3, 0) = zero;
                v[0] = vt[0];
                v[1] = vt[1];
                v[2] = vt[2];
            }
        }
    }

} // lapack

#endif // __MOVE_BULGE_HH__
