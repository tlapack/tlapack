/// @file nrm2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
///
/// Anderson E. (2017)
/// Algorithm 978: Safe Scaling in the Level 1 BLAS
/// ACM Trans Math Softw 44:1--28
/// @see https://doi.org/10.1145/3061665
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_NRM2_HH
#define TLAPACK_BLAS_NRM2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/lassq.hpp"

namespace tlapack {

/**
 * @return 2-norm of vector,
 *     $|| x ||_2 := (\sum_{i=0}^{n-1} |x_i|^2)^{1/2}$.
 *
 * @param[in] x A n-element vector.
 *
 * @ingroup blas1
 */
template <TLAPACK_VECTOR vector_t, disable_if_allow_optblas_t<vector_t> = 0>
inline auto nrm2(const vector_t& x)
{
    using real_t = real_type<type_t<vector_t> >;

    // scaled sum of squares
    real_t scl(1);
    real_t sumsq(0);
    lassq(x, scl, sumsq);

    return real_t(scl * sqrt(sumsq));
}

#ifdef USE_LAPACKPP_WRAPPERS

template <TLAPACK_VECTOR vector_t, enable_if_allow_optblas_t<vector_t> = 0>
inline auto nrm2(vector_t const& x)
{
    // Legacy objects
    auto x_ = legacy_vector(x);

    // Constants to forward
    const auto& n = x_.n;

    return ::blas::nrm2(n, x_.ptr, x_.inc);
}

#endif

}  // namespace tlapack

#endif  // #ifndef TLAPACK_BLAS_NRM2_HH
