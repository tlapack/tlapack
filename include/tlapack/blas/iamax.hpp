/// @file iamax.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_IAMAX_HH
#define TLAPACK_BLAS_IAMAX_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Options for iamax.
 *
 * Initialize using a lambda function (C++17 or higher):
 * ```c++
 * IamaxOpts opts( [](const T& x) { return my_abs(x); } );
 * ```
 * or using a functor:
 * ```c++
 * struct abs_f {
 *    inline constexpr real_type<T> operator()(const T& x) const {
 *       return my_abs(x);
 *   }
 * };
 * abs_f absf;
 * IamaxOpts<abs_f> opts( absf );
 * ```
 */
template <class abs_f>
struct IamaxOpts : public EcOpts {
    inline constexpr IamaxOpts(abs_f absf, const EcOpts& opts = {})
        : EcOpts(opts), absf(absf){};

    abs_f absf;  ///< Absolute value function
                 ///< In reference BLAS, absf(a) := |Re(a)| + |Im(a)|
};

/**
 * @brief Return $\arg\max_{i=0}^{n-1} |x_i|$
 *
 * Version with NaN checks.
 * @see iamax_nc(const vector_t& x, abs_f absf) for the version that does not
 * check for NaNs.
 *
 * @param[in] x The n-element vector x.
 *
 * @param[in] absf Absolute value function.
 *      In reference BLAS, absf(a) := |Re(a)| + |Im(a)|.
 *      We also use |a| to denote absf(a).
 *
 * @return In priority order:
 * 1. 0 if n <= 0,
 * 2. the index of the first `NAN` in $x$ if it exists,
 * 3. the index of the first `Infinity` in $x$ if it exists,
 * 4. the Index of the infinity-norm of $x$, $|| x ||_{inf}$,
 *     $\arg\max_{i=0}^{n-1} |x_i|$.
 *
 * @ingroup blas1
 */
template <TLAPACK_VECTOR vector_t, class abs_f>
size_type<vector_t> iamax_ec(const vector_t& x, abs_f absf)
{
    // data traits
    using idx_t = size_type<vector_t>;
    using T = type_t<vector_t>;
    using real_t = real_type<T>;

    // constants
    const real_t oneFourth(0.25);
    const idx_t n = size(x);

    // quick return
    if (n <= 0) return 0;

    bool scaledsmax = false;  // indicates whether |x_i| = Inf
    real_t smax(-1);
    idx_t index = -1;
    idx_t i = 0;
    for (; i < n; ++i) {
        if (isnan(x[i])) {
            // return when first NaN found
            return i;
        }
        else if (isinf(x[i])) {
            // keep looking for first NaN
            for (idx_t k = i + 1; k < n; ++k) {
                if (isnan(x[k])) {
                    // return when first NaN found
                    return k;
                }
            }

            // return the position of the first Inf
            return i;
        }
        else {  // still no Inf found yet
            if (is_real<T>) {
                real_t a = absf(x[i]);
                if (a > smax) {
                    smax = a;
                    index = i;
                }
            }
            else if (!scaledsmax) {  // no |x_i| = Inf  yet
                real_t a = absf(x[i]);
                if (isinf(a)) {
                    scaledsmax = true;
                    smax = absf(oneFourth * x[i]);
                    index = i;
                }
                else if (a > smax) {
                    smax = a;
                    index = i;
                }
            }
            else {  // scaledsmax = true
                real_t a = absf(oneFourth * x[i]);
                if (a > smax) {
                    smax = a;
                    index = i;
                }
            }
        }
    }

    return index;
}

/**
 * @brief Return $\arg\max_{i=0}^{n-1} |x_i|$
 *
 * Version with no NaN checks.
 * @see iamax_ec(const vector_t& x, abs_f absf) for the version that check for
 * NaNs.
 *
 * @param[in] x The n-element vector x.
 *
 * @param[in] absf Absolute value function.
 *      In reference BLAS, absf(a) := |Re(a)| + |Im(a)|
 *      We also use |a| to denote absf(a).
 *
 * @return In priority order:
 * 1. 0 if n <= 0,
 * 2. the index of the first `Infinity` in $x$ if it exists,
 * 3. the Index of the infinity-norm of $x$, $|| x ||_{inf}$,
 *     $\arg\max_{i=0}^{n-1} |x_i|$.
 *
 * @ingroup blas1
 */
template <TLAPACK_VECTOR vector_t, class abs_f>
size_type<vector_t> iamax_nc(const vector_t& x, abs_f absf)
{
    // data traits
    using idx_t = size_type<vector_t>;
    using T = type_t<vector_t>;
    using real_t = real_type<T>;

    // constants
    const real_t oneFourth(0.25);
    const idx_t n = size(x);

    // quick return
    if (n <= 0) return 0;

    bool scaledsmax = false;  // indicates whether |x_i| = Inf
    real_t smax(-1);
    idx_t index = -1;
    idx_t i = 0;
    for (; i < n; ++i) {
        if (isinf(x[i])) {
            // return the position of the first Inf
            return i;
        }
        else {  // still no Inf found yet
            if (is_real<T>) {
                real_t a = absf(x[i]);
                if (a > smax) {
                    smax = a;
                    index = i;
                }
            }
            else if (!scaledsmax) {  // no |x_i| = Inf  yet
                real_t a = absf(x[i]);
                if (isinf(a)) {
                    scaledsmax = true;
                    smax = absf(oneFourth * x[i]);
                    index = i;
                }
                else if (a > smax) {
                    smax = a;
                    index = i;
                }
            }
            else {  // scaledsmax = true
                real_t a = absf(oneFourth * x[i]);
                if (a > smax) {
                    smax = a;
                    index = i;
                }
            }
        }
    }

    return (index != idx_t(-1)) ? index : 0;
}

/**
 * @brief Return $\arg\max_{i=0}^{n-1} |x_i|$
 *
 * @see iamax_nc( const vector_t& x, abs_f absf ) for the version that
 * does not check for NaNs.
 * @see iamax_ec( const vector_t& x, abs_f absf ) for the version that
 * check for NaNs.
 *
 * @param[in] x The n-element vector x.
 *
 * @param[in] opts Options.
 *      Define the behavior of checks for NaNs.
 *      Also define the absolute value function.
 *      - Default NaN check controlled by TLAPACK_DEFAULT_NANCHECK
 *      - Default absolute value function as in Reference BLAS,
 *          |a| := |Re(a)| + |Im(a)|.
 *
 * @ingroup blas1
 */
template <TLAPACK_VECTOR vector_t, class abs_f>
inline size_type<vector_t> iamax(const vector_t& x,
                                 const IamaxOpts<abs_f>& opts)
{
    return (opts.ec.nan == true) ? iamax_ec(x, opts.absf)
                                 : iamax_nc(x, opts.absf);
}

template <TLAPACK_VECTOR vector_t, disable_if_allow_optblas_t<vector_t> = 0>
inline size_type<vector_t> iamax(const vector_t& x)
{
    using T = type_t<vector_t>;
    using real_t = real_type<T>;

#if __cplusplus >= 201703L
    IamaxOpts opts([](const T& x) -> real_t { return abs1(x); });
#else
    struct abs_f {
        inline constexpr real_t operator()(const T& x) const { return abs1(x); }
    };
    abs_f absf;
    IamaxOpts<abs_f> opts(absf);
#endif

    return iamax(x, opts);
}

#ifdef TLAPACK_USE_LAPACKPP

template <TLAPACK_LEGACY_VECTOR vector_t,
          enable_if_allow_optblas_t<vector_t> = 0>
inline size_type<vector_t> iamax(vector_t const& x)
{
    // Legacy objects
    auto x_ = legacy_vector(x);

    // Constants to forward
    const auto& n = x_.n;

    return ::blas::iamax(n, x_.ptr, x_.inc);
}

#endif

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_IAMAX_HH
