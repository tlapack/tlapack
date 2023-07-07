/// @file example_gehd2.cpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>

// <T>LAPACK
#include <tlapack/lapack/gehrd.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/multishift_qr.hpp>
#include <tlapack/lapack/unghr.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
inline void printMatrix(const matrix_t& A)
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

class rand_generator {
   private:
    const uint64_t a = 6364136223846793005;
    const uint64_t c = 1442695040888963407;
    uint64_t state = 1302;

   public:
    uint32_t min() { return 0; }

    uint32_t max() { return UINT32_MAX; }

    void seed(uint64_t s) { state = s; }

    uint32_t operator()()
    {
        state = state * a + c;
        return state >> 32;
    }
};

template <typename T>
T rand_helper(rand_generator& gen)
{
    return static_cast<T>(gen()) / static_cast<T>(gen.max());
}

extern "C" {
void fortran_slaqr0(const bool* wantt,
                    const bool* wantz,
                    const int* n,
                    const int* ilo,
                    const int* ihi,
                    float* H,
                    const int* ldh,
                    float* wr,
                    float* wi,
                    float* Z,
                    const int* ldz,
                    int* info,
                    int* n_aed,
                    int* n_sweep,
                    int* n_shifts);
void fortran_sgehrd(const int* n,
                    const int* ilo,
                    const int* ihi,
                    float* H,
                    const int* ldh,
                    float* tau,
                    int* info);
void fortran_sorghr(const int* n,
                    const int* ilo,
                    const int* ihi,
                    float* H,
                    const int* ldh,
                    float* tau,
                    int* info);
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t n,
         int seed,
         bool use_fortran_gehrd_unghr = false,
         bool use_fortran_hqr = false,
         bool verbose = false)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using std::size_t;

    constexpr bool use_lapack
#ifdef USE_LAPACK
        = std::is_same_v<T, float>;
#else
        = false;
#endif

    // Functor for creating new matrices of type matrix_t
    tlapack::Create<matrix_t> new_matrix;

    rand_generator gen;
    gen.seed(seed);

    // Constants
    int one = 1, len = n;
    bool yes = true;

    // Arrays
    std::vector<T> tau(n);

    // Matrix views
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> H_;
    auto H = new_matrix(H_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, n, n);

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            A(i, j) = static_cast<float>(0xDEADBEEF);
            Q(i, j) = static_cast<float>(0xCAFED00D);
        }
        for (size_t i = 0; i < n; ++i) {
            H(i, j) = static_cast<float>(0xFEE1DEAD);
        }
        tau[j] = static_cast<float>(0xFFBADD11);
    }

    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            A(i, j) = rand_helper<T>(gen);

    // Frobenius norm of A
    auto normA = tlapack::lange(tlapack::frob_norm, A);

    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(A);
    }

    // Copy A to Q
    tlapack::lacpy(tlapack::Uplo::General, A, Q);

    // 1) Compute A = QHQ* (Stored in the matrix Q)

    // Record start time
    auto startQHQ = std::chrono::high_resolution_clock::now();
    {
        // Hessenberg factorization
        int info;
        if constexpr (use_lapack) {
            if (use_fortran_gehrd_unghr)
                fortran_sgehrd(&len, &one, &len, Q.ptr, &len, tau.data(),
                               &info);
            else
                info = tlapack::gehrd(0, n, Q, tau);
        }
        else
            info = tlapack::gehrd(0, n, Q, tau);
        if (info != 0)
            std::cout << "gehrd ended with info " << info << std::endl;
    }
    // Record end time
    auto endQHQ = std::chrono::high_resolution_clock::now();

    // Save the H matrix
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < std::min(n, j + 2); ++i)
            H(i, j) = Q(i, j);

    // Record start time
    auto startQ = std::chrono::high_resolution_clock::now();
    {
        // Generate Q = H_1 H_2 ... H_n
        int info;
        if constexpr (use_lapack) {
            if (use_fortran_gehrd_unghr)
                fortran_sorghr(&len, &one, &len, Q.ptr, &len, tau.data(),
                               &info);
            else
                info = tlapack::unghr(0, n, Q, tau);
        }
        else
            info = tlapack::unghr(0, n, Q, tau);
        if (info != 0)
            std::cout << "unghr ended with info " << info << std::endl;
    }
    // Record end time
    auto endQ = std::chrono::high_resolution_clock::now();

    // Remove junk from lower half of H
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            H(i, j) = 0.0;

    // Record start time
    int info = 0;
    int n_aed = 0;
    int n_sweep = 0;
    int n_shifts_total = 0;
    auto startSchur = std::chrono::high_resolution_clock::now();
    {
        // Shur factorization

        if constexpr (use_lapack) {
            if (use_fortran_hqr) {
                std::vector<T> wr(n);
                std::vector<T> wi(n);
                fortran_slaqr0(&yes, &yes, &len, &one, &len, H.ptr, &len,
                               wr.data(), wi.data(), Q.ptr, &len, &info, &n_aed,
                               &n_sweep, &n_shifts_total);
            }
            else {
                tlapack::francis_opts_t<> opts;
                std::vector<std::complex<real_t>> w(n);
                info = tlapack::multishift_qr(true, true, 0, n, H, w, Q, opts);
                n_aed = opts.n_aed;
                n_sweep = opts.n_sweep;
                n_shifts_total = opts.n_shifts_total;
            }
        }
        else {
            tlapack::francis_opts_t<> opts;
            std::vector<std::complex<real_t>> w(n);
            info = tlapack::multishift_qr(true, true, 0, n, H, w, Q, opts);
            n_aed = opts.n_aed;
            n_sweep = opts.n_sweep;
            n_shifts_total = opts.n_shifts_total;
        }
    }
    // Record end time
    auto endSchur = std::chrono::high_resolution_clock::now();

    if (info != 0)
        std::cout << "multishift_qr ended with info " << info << std::endl;

    // Remove junk from lower half of H
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            H(i, j) = 0.0;

    // Compute elapsed time in nanoseconds
    auto elapsedQHQ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endQHQ - startQHQ);
    auto elapsedQ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endQ - startQ);
    auto elapsedSchur = std::chrono::duration_cast<std::chrono::nanoseconds>(
        endSchur - startSchur);

    // Print Q and H
    if (verbose) {
        std::cout << std::endl << "Q = ";
        printMatrix(Q);
        std::cout << std::endl << "H = ";
        printMatrix(H);
    }

    real_t norm_orth_1, norm_repres_1;

    // 2) Compute ||Q'Q - I||_F

    {
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        tlapack::laset(tlapack::Uplo::General, (T)0.0, (T)1.0, work);
        // work receives Q'Q - I
        // tlapack::syrk( tlapack::Uplo::Upper, tlapack::Op::ConjTrans, (T) 1.0,
        // Q, (T) -1.0, work );
        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans, (T)1.0, Q,
                      Q, (T)-1.0, work);

        // Compute ||Q'Q - I||_F
        norm_orth_1 =
            tlapack::lansy(tlapack::frob_norm, tlapack::Uplo::Upper, work);

        if (verbose) {
            std::cout << std::endl << "Q'Q-I = ";
            printMatrix(work);
        }
    }

    // 3) Compute ||QHQ* - A||_F / ||A||_F

    std::vector<T> Hcopy_;
    auto H_copy = new_matrix(Hcopy_, n, n);
    tlapack::lacpy(tlapack::Uplo::General, H, H_copy);
    {
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABC);

        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, (T)1.0, Q, H,
                      work);
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans, (T)1.0,
                      work, Q, H);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                H(i, j) -= A(i, j);

        if (verbose) {
            std::cout << std::endl << "QHQ'-A = ";
            printMatrix(H);
        }

        norm_repres_1 = tlapack::lange(tlapack::frob_norm, H) / normA;
    }

    // 4) Compute Q*AQ (usefull for debugging)

    if (verbose) {
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABC);

        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans, (T)1.0, Q,
                      A, work);
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, (T)1.0, work,
                      Q, A);

        std::cout << std::endl << "Q'AQ = ";
        printMatrix(A);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                A(i, j) -= H_copy(i, j);

        std::cout << std::endl << "Q'AQ - H = ";
        printMatrix(A);
    }

    std::cout << std::endl;
    std::cout << "Hessenberg time = " << elapsedQHQ.count() * 1.0e-9 << " s"
              << std::endl;
    std::cout << "Q forming time = " << elapsedQ.count() * 1.0e-9 << " s"
              << std::endl;
    std::cout << "QR time = " << elapsedSchur.count() * 1.0e-9 << " s"
              << std::endl;
    std::cout << "||QHQ* - A||_F/||A||_F  = " << norm_repres_1
              << ",        ||Q'Q - I||_F  = " << norm_orth_1 << std::endl;
    std::cout << "AED calls   " << n_aed << std::endl
              << "Sweep calls " << n_sweep << std::endl
              << "Shifts used " << n_shifts_total << std::endl;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int n, seed;
    bool use_fortran_gehrd_unghr;
    bool use_fortran_hqr;
    bool verbose;

    // Default arguments
    n = (argc <= 1) ? 100 : atoi(argv[1]);
    seed = (argc <= 2) ? 1302 : atoi(argv[2]);
    use_fortran_gehrd_unghr = (argc <= 3) ? false : atoi(argv[3]);
    use_fortran_hqr = (argc <= 4) ? false : atoi(argv[4]);
    verbose = (argc <= 5) ? false : atoi(argv[5]);

    // Usage
    if (argc > 6) {
        std::cout << "Usage: " << argv[0]
                  << " [n] [seed] [use_fortran_gehrd_unghr] [use_fortran_hqr] "
                     "[verbose]"
                  << std::endl
                  << "  n: matrix size (default: 100)" << std::endl
                  << "  seed: random seed (default: 1302)" << std::endl
                  << "  use_fortran_gehrd_unghr: use fortran gehrd/unghr "
                     "(default: false (0))"
                  << std::endl
                  << "  use_fortran_hqr: use fortran hqr (default: false "
                     "(0))"
                  << std::endl
                  << "  verbose: verbose output (default: false (0))"
                  << std::endl;
        return 1;
    }

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float >( %d )", n);
    run<float>(n, seed, use_fortran_gehrd_unghr, use_fortran_hqr, verbose);
    printf("-----------------------\n");

    printf("run< std::complex<float>  >( %d )", n);
    run<std::complex<float>>(n, seed, use_fortran_gehrd_unghr, use_fortran_hqr,
                             verbose);
    printf("-----------------------\n");

    printf("run< double >( %d )", n);
    run<double>(n, seed, use_fortran_gehrd_unghr, use_fortran_hqr, verbose);
    printf("-----------------------\n");

    printf("run< std::complex<double>  >( %d )", n);
    run<std::complex<double>>(n, seed, use_fortran_gehrd_unghr, use_fortran_hqr,
                              verbose);
    printf("-----------------------\n");

    printf("run< long double >( %d )", n);
    run<long double>(n, seed, use_fortran_gehrd_unghr, use_fortran_hqr,
                     verbose);
    printf("-----------------------\n");

    printf("run< std::complex<long double>  >( %d )", n);
    run<std::complex<long double>>(n, seed, use_fortran_gehrd_unghr,
                                   use_fortran_hqr, verbose);
    printf("-----------------------\n");

    return 0;
}
