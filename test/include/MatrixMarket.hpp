/// @file MatrixMarket.hpp
/// @brief MaxtrixMarket class and random generators.
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MATRIXMARKET_HH
#define TLAPACK_MATRIXMARKET_HH

#include <random>
#include <tlapack/base/utils.hpp>
#include <type_traits>

namespace tlapack {

/**
 * @brief Permuted Congruential Generator
 *
 * Defined in https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf as PCG-XSL-RR.
 * Constants taken from https://github.com/imneme/pcg-cpp.
 */
class PCG32 {
   private:
    const uint64_t a = 6364136223846793005ULL;  ///< multiplier (64-bit)
    const uint64_t c = 1442695040888963407ULL;  ///< increment (64-bit)
    uint64_t state;                             ///< RNG state (64-bit)

   public:
    /// @brief Constructor
    /// @param s Default is 1302 for no good reason.
    PCG32(uint64_t s = 1302) noexcept { seed(s); }

    /// Sets the current state of PCG32. Same as PCG32(s).
    void seed(uint64_t s) noexcept
    {
        state = 0;
        operator()();
        state += s;
        operator()();
    }

    static constexpr uint32_t min() noexcept { return 0; }
    static constexpr uint32_t max() noexcept { return UINT32_MAX; }

    /// Generates a pseudo-random number using PCG's output function (XSH-RR).
    uint32_t operator()() noexcept
    {
        // Constants for a 64-bit state and 32-bit output
        const uint8_t bits = 64;
        const uint8_t opbits = 5;
        constexpr uint8_t mask = (1 << opbits) - 1;
        constexpr uint8_t xshift = (opbits + 32) / 2;
        constexpr uint8_t bottomspare = (bits - 32) - opbits;

        // Random rotation
        uint8_t rot = uint8_t(state >> (bits - opbits)) & mask;

        // XOR shift high
        uint32_t result =
            static_cast<uint32_t>((state ^ (state >> xshift)) >> bottomspare);

        // Rotate right
        result = (result >> rot) | (result << ((-rot) & mask));

        // Updates the state
        state = state * a + c;

        return result;
    }
};

/**
 * @brief Helper function to generate random numbers.
 * @param[in,out] gen Random number generator.
 * @param[in,out] d Random distribution.
 */
template <class T,
          class Generator,
          class Distribution,
          enable_if_t<is_real<T>, bool> = true>
T rand_helper(Generator& gen, Distribution& d)
{
    return T(d(gen));
}

/**
 * @overload T rand_helper(Generator& gen, Distribution& d)
 */
template <class T,
          class Generator,
          class Distribution,
          enable_if_t<is_complex<T>, bool> = true>
T rand_helper(Generator& gen, Distribution& d)
{
    using real_t = real_type<T>;
    real_t r1(d(gen));
    real_t r2(d(gen));
    return complex_type<real_t>(r1, r2);
}

/**
 * @brief Helper function to generate random numbers using an uniform
 * distribution in [0,1).
 *
 * The type of the uniform distribution is T if real_type<T> is either float,
 * double or long double. Otherwise, the uniform distribution will be defined
 * using the type float.
 *
 * @param[in,out] gen Random number generator.
 */
template <class T, class Generator>
T rand_helper(Generator& gen)
{
    using real_t = real_type<T>;
    using dist_t =
        typename std::conditional<(is_same_v<real_t, float> ||
                                   is_same_v<real_t, double> ||
                                   is_same_v<real_t, long double>),
                                  std::normal_distribution<real_t>,
                                  std::normal_distribution<float> >::type;

    dist_t d;
    return rand_helper<T>(gen, d);
}

/**
 * @brief MatrixMarket class.
 *
 * This class provides methods to read matrices from files and to generate
 * random or structured matrices.
 */
struct MatrixMarket {
    /**
     * @brief Read a dense matrix from an input stream (file, stdin, etc).
     *
     * The data is read in column-major format.
     *
     * @param[out] A Matrix.
     * @param[in,out] is Input stream.
     */
    template <TLAPACK_MATRIX matrix_t>
    void colmajor_read(matrix_t& A, std::istream& is) const
    {
        using idx_t = size_type<matrix_t>;

        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                is >> A(i, j);
    }

    /**
     * @brief Generate a random dense matrix.
     *
     * @param[out] A Matrix.
     */
    template <TLAPACK_MATRIX matrix_t>
    void random(matrix_t& A)
    {
        using T = type_t<matrix_t>;
        using idx_t = size_type<matrix_t>;

        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                A(i, j) = rand_helper<T>(gen);
    }

    /**
     * @brief Generate an upper- or lower-triangular random matrix.
     *
     * Put a single garbage value float(0xCAFEBABE) in the opposite triangle.
     *
     * @param[in] uplo Upper or lower triangular.
     * @param[out] A Matrix.
     */
    template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t>
    void random(uplo_t uplo, matrix_t& A)
    {
        using T = type_t<matrix_t>;
        using idx_t = size_type<matrix_t>;

        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        if (uplo == Uplo::Upper) {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < m; ++i)
                    if (i <= j)
                        A(i, j) = rand_helper<T>(gen);
                    else
                        A(i, j) = T(float(0xCAFEBABE));
        }
        else {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < m; ++i)
                    if (i >= j)
                        A(i, j) = rand_helper<T>(gen);
                    else
                        A(i, j) = T(float(0xCAFEBABE));
        }
    }

    /**
     * @brief Generate a random upper Hessenberg matrix.
     *
     * Put a single garbage value float(0xFA57C0DE) below the first subdiagonal.
     *
     * @param[out] A Matrix.
     */
    template <TLAPACK_MATRIX matrix_t>
    void hessenberg(matrix_t& A)
    {
        using T = type_t<matrix_t>;
        using idx_t = size_type<matrix_t>;

        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                if (i <= j + 1)
                    A(i, j) = rand_helper<T>(gen);
                else
                    A(i, j) = T(float(0xFA57C0DE));
    }

    /**
     * @brief Generate a Hilbert matrix.
     *
     * The Hilbert matrix is defined as:
     *
     * \[
     *    A_{ij} = \frac{1}{i + j + 1}
     * \]
     *
     * @param[out] A Matrix.
     */
    template <TLAPACK_MATRIX matrix_t>
    void hilbert(matrix_t& A) const
    {
        using T = type_t<matrix_t>;
        using idx_t = size_type<matrix_t>;

        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                A(i, j) = T(1) / T(i + j + 1);
    }

    /**
     * @brief Generate a matrix with a single value in all entries.
     *
     * @param[out] A Matrix.
     * @param[in] val Value.
     */
    template <TLAPACK_MATRIX matrix_t>
    void single_value(matrix_t& A, const type_t<matrix_t>& val) const
    {
        using idx_t = size_type<matrix_t>;

        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                A(i, j) = val;
    }

    /**
     * @brief Generate an upper- or lower-triangular matrix with a single value
     * in all entries.
     *
     * Put a single garbage value float(0xCAFEBABE) in the opposite triangle.
     *
     * @param[in] uplo Upper or lower triangular.
     * @param[out] A Matrix.
     * @param[in] val Value.
     */
    template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t>
    void single_value(uplo_t uplo,
                      matrix_t& A,
                      const type_t<matrix_t>& val) const
    {
        using T = type_t<matrix_t>;
        using idx_t = size_type<matrix_t>;

        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        if (uplo == Uplo::Upper) {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < m; ++i)
                    if (i <= j)
                        A(i, j) = val;
                    else
                        A(i, j) = T(float(0xCAFEBABE));
        }
        else {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < m; ++i)
                    if (i >= j)
                        A(i, j) = val;
                    else
                        A(i, j) = T(float(0xCAFEBABE));
        }
    }

    PCG32 gen;
};

}  // namespace tlapack

#endif  // TLAPACK_MATRIXMARKET_HH