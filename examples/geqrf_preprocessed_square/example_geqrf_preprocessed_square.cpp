/// @file example_geqrf_preprocessed_square.cpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
//
// Copyright (c) 2026, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// Example: solve a consistent linear system by
// 1) building a Householder reflector for the right-hand side vector b, such
// that H b = [ 0 0 ... 0 *]ᴴ, where the modulus of * is the 2-norm of b,
// 2) applying that reflector to A to preprocess the system,
// 3) computing an LQ factorization of the transformed matrix HA
// 4) and using the LQ factors to obtain the solution.
//
// This example follows Golub and Van Loan, Matrix Computations, 4th edition,
// section 5.6.1, which discusses consistent systems and transforms that
// preserve solvability. However, we use the LQ factorization instead of the QR
// factorization. We quote:
//
// If the QR factorization is used to solve Ax=b, then we ordinarily have to
// carry out a backsubstitution: Rx = Qᴴb. However, this can be avoided by
// "preprocessing" b. Suppose H is a Householder matrix such that H b = β eₙ
// where eₙ is the last column of Iₙ. If we compute the QR factorization of
// (HA)ᴴ, then A = HᴴRᴴQᴴ and the system transform to Rᴴy = β eₙ where y = Qᴴx.
// Since Rᴴ is lower triangular, y = (β/conj(rₙₙ))eₙ and so y = (β/conj(rₙₙ))
// Q(:,n).

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/gemv.hpp>
#include <tlapack/blas/scal.hpp>
#include <tlapack/lapack/gelqf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/larf.hpp>
#include <tlapack/lapack/larfg.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/unmlq.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

/// Print vector v in the standard output
template <typename T>
void printVector(const std::vector<T>& v)
{
    using idx_t = typename std::vector<T>::size_type;
    const idx_t n = v.size();
    for (idx_t i = 0; i < n; ++i) {
        std::cout << std::endl << v[i] << " ";
    }
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t n)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;
    using tlapack::conj;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if n is large
    bool verbose = false;

    // Arrays
    std::vector<T> tau(n);
    std::vector<T> b(n);
    std::vector<T> b_orig(n);

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> A_orig_;
    auto A_orig = new_matrix(A_orig_, n, n);
    std::vector<T> x_;
    auto x = new_matrix(x_, n, 1);

    // Initialize arrays with junk
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            if constexpr (tlapack::is_complex<T>)
                A(i, j) = T(static_cast<float>(0xDEADBEEF),
                            static_cast<float>(0xDEADBEEF));
            else
                A(i, j) = T(static_cast<float>(0xDEADBEEF));
        }
        if constexpr (tlapack::is_complex<T>) {
            tau[j] = T(static_cast<float>(0xFFBADD11),
                       static_cast<float>(0xFFBADD11));
            x_[j] = T(static_cast<float>(0xBAADF00D),
                      static_cast<float>(0xBAADF00D));
        }
        else {
            tau[j] = T(static_cast<float>(0xFFBADD11));
            x_[j] = T(static_cast<float>(0xBAADF00D));
        }
    }

    // Generate a random matrix in A
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            if constexpr (tlapack::is_complex<T>)
                A(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            else
                A(i, j) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));

    // Copy A to A_orig
    tlapack::lacpy(tlapack::GENERAL, A, A_orig);

    // Generate a random vector in x
    for (idx_t j = 0; j < n; ++j)
        if constexpr (tlapack::is_complex<T>)
            x_[j] =
                T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                  static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        else
            x_[j] =
                T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));

    // Generate b = A*x so that the system is consistent.
    tlapack::gemv(tlapack::Op::NoTrans, static_cast<T>(1.0), A, x_,
                  static_cast<T>(0.0), b);

    // Copy b to b_orig
    for (idx_t i = 0; i < n; ++i)
        b_orig[i] = b[i];

    // Compute || b ||₂
    auto norm_b = tlapack::nrm2(b);

    // Print A and b
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(A_orig);
        std::cout << std::endl;

        std::cout << std::endl << "b = ";
        printVector(b_orig);
        std::cout << std::endl;
    }

    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        // 1) Compute the Householder reflector for b
        //    6*m FLOPS
        T tau_b = static_cast<T>(0.0);
        tlapack::larfg(tlapack::Direction::Backward,
                       tlapack::StoreV::Columnwise, b, tau_b);

        // 2) Apply the Householder reflector to A to produce HA
        //    4*n² FLOPS
        tlapack::larf(tlapack::Side::Left, tlapack::Direction::Backward,
                      tlapack::StoreV::Columnwise, b, conj(tau_b), A);

        // 3) Compute the LQ factorization of HA
        //    4/3*n³ FLOPS
        tlapack::gelqf(A, tau);

        // 4) Compute the last row of Q by applying the Householder
        //    reflectors to the last row of the identity matrix

        // 4.1) Set x to be the last row of the identity matrix
        for (idx_t i = 0; i < n; ++i)
            x_[i] = static_cast<T>(0.0);
        x_[n - 1] = static_cast<T>(1.0);

        // 4.2) Apply the Householder reflectors to x
        //      4*n² FLOPS
        tlapack::unmlq(tlapack::Side::Left, tlapack::Op::ConjTrans, A, tau, x);

        // 5) Scale by β/lₙₙ. This is the solution to the system.
        //    n FLOPS
        T scale = b[n - 1] / A(n - 1, n - 1);
        tlapack::scal(scale, x_);
    }

    // Record end time
    auto endQR = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedQR =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endQR - startQR);

    // Print x
    if (verbose) {
        std::cout << std::endl << "x = ";
        printVector(x_);
        std::cout << std::endl;
    }

    // Compute FLOPS
    // 6m + 4n² + 4/3*n³ + 4n² + n = 4/3*n³ + 8n² + 6m
    double flopsQR =
        (4.0e+00 / 3.0e+00 * ((double)n) * ((double)n) * ((double)n) +
         8.0e+00 * ((double)n) * ((double)n) + 6.0e+00 * ((double)n)) /
        (elapsedQR.count() * 1.0e-9);

    // 6) Check || b - A * x ||₂ / (|| b ||₂ + || A ||ꜰ * || x ||₂)

    // 6.1) Compute || x ||₂
    auto norm_x = tlapack::nrm2(x_);

    // 6.2) Compute || b - A * x ||₂
    tlapack::gemv(tlapack::Op::NoTrans, static_cast<T>(-1.0), A_orig, x_,
                  static_cast<T>(1.0), b_orig);
    auto norm_residual = tlapack::nrm2(b_orig);

    // 6.3) Compute || A ||ꜰ
    auto norm_A = tlapack::lange(tlapack::FROB_NORM, A_orig);

    auto check = norm_residual / (norm_b + norm_A * norm_x);

    // Output

    std::cout << std::endl;
    std::cout << "time = " << elapsedQR.count() * 1.0e-6 << " ms"
              << ",   GFlop/sec = " << flopsQR * 1.0e-9;
    std::cout << std::endl;
    std::cout << "|| b - A * x ||₂ / (|| b ||₂ + || A ||ꜰ * || x ||₂) = "
              << check << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int n;

    // Default arguments
    n = (argc < 3) ? 199 : atoi(argv[2]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d )", n);
    run<float>(n);
    printf("-----------------------\n");

    printf("run< double >( %d )", n);
    run<double>(n);
    printf("-----------------------\n");

    printf("run< long double >( %d )", n);
    run<long double>(n);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d )", n);
    run<std::complex<float>>(n);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d )", n);
    run<std::complex<double>>(n);
    printf("-----------------------\n");

    printf("run< complex<long double> >( %d )", n);
    run<std::complex<long double>>(n);
    printf("-----------------------\n");

    return 0;
}
