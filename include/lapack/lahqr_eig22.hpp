/// @file lahqr_eig22.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LAHQR_EIG22_HH__
#define __LAHQR_EIG22_HH__

#include <complex>

#include "lapack/utils.hpp"
#include "lapack/types.hpp"

namespace lapack
{

    /** Computes the eigenvalues of a 2x2 matrix A
     *
     * @param[in] a00
     *      Element (0,0) of A.
     * @param[in] a01
     *      Element (0,1) of A.
     * @param[in] a10
     *      Element (1,0) of A.
     * @param[in] a11
     *      Element (1,1) of A.
     * @param[out] s1
     * @param[out] s2
     *      s1 and s2 are the eigenvalues of A
     *
     * @ingroup geev
     */
    template <
        typename T,
        typename real_t = real_type<T>>
    void lahqr_eig22(T a00, T a01, T a10, T a11, std::complex<real_t> &s1, std::complex<real_t> &s2)
    {

        // Using
        using blas::abs1;
        using blas::real;

        // Constants
        const real_t rzero(0);
        const real_t two(2);
        const T zero(0);

        auto s = abs1(a00) + abs1(a01) + abs1(a10) + abs1(a11);
        if (s == rzero)
        {
            s1 = zero;
            s2 = zero;
            return;
        }

        a00 = a00 / s;
        a01 = a01 / s;
        a10 = a10 / s;
        a11 = a11 / s;
        auto tr = (a00 + a11) / two;
        std::complex<real_t> det = (a00 - tr) * (a00 - tr) + a01 * a10;
        auto rtdisc = sqrt(det);

        s1 = s*(tr + rtdisc);
        s2 = s*(tr - rtdisc);
    }

} // lapack

#endif // __LAHQR_EIG22_HH__
