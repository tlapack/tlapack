/// @file example_trmm_mixed.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <chrono>  // for high_resolution_clock
#include <set>     // For sets

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/eigen_bfloat16.hpp>
#include <tlapack/plugins/legacyArray.hpp>

// Plugin for debug
#include <tlapack/plugins/debugutils.hpp>

// <T>LAPACK
#include <tlapack/base/utils.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lantr.hpp>
#include <tlapack/lapack/trmm_blocked_mixed.hpp>

#ifdef TLAPACK_USE_MKL
    #include <mkl.h>
    #include <mkl_cblas.h>
#endif

using namespace tlapack;

// Generate m-by-n upper-triangle random matrix
template <class matrix_t>
void gen_matrixA(matrix_t& A)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    const int m = nrows(A);
    const int n = ncols(A);
    for (idx_t i = 0; i < m; ++i)
        for (idx_t j = 0; j < n; ++j)
            if (i <= j)
                A(i, j) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));
            else
                A(i, j) = T(float(0xCAFEBABE));
}

// Generate m-by-n random matrix
template <class matrix_t>
void gen_matrixB(matrix_t& B)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    const int m = nrows(B);
    const int n = ncols(B);
    for (idx_t i = 0; i < m; ++i)
        for (idx_t j = 0; j < n; ++j)
            B(i, j) =
                T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
}

template <class TestType>
void run(int nruns, int m, int n, int nb = 128, bool comparePerformance = true)
{
    using T1 = TestType::first_type;
    using T2 = TestType::second_type;

    using matrix1_t =
        tlapack::LegacyMatrix<T1, std::size_t, tlapack::Layout::ColMajor>;
    using matrix2_t =
        tlapack::LegacyMatrix<T2, std::size_t, tlapack::Layout::ColMajor>;

    using idx_t = size_type<matrix1_t>;
    typedef real_type<T1> real_t;

    // FLOPs
    const double flopTrmm = n * double(m) * double(m);

    // Functor
    Create<matrix1_t> new_matrix1;
    Create<matrix2_t> new_matrix2;

    std::vector<T2> A_;
    auto A = new_matrix2(A_, m, m);
    std::vector<T1> B_;
    auto B = new_matrix1(B_, m, n);
    std::vector<T1> E_;
    auto E = new_matrix1(E_, m, n);
    std::vector<T2> W_;
    auto W = new_matrix2(W_, min(nb, m), n);

    std::cout << "nruns = " << nruns << std::endl;
    std::cout << "nb = " << nb << std::endl;
    std::cout << "m = " << m << std::endl;
    std::cout << "n = " << n << std::endl;

    double mintime = 999999 * 1e9;  // in nanoseconds
    for (int r = 0; r < nruns; ++r) {
        // Generate m-by-m upper-triangle random matrix
        gen_matrixA(A);

        // Generate m-by-n random matrix
        gen_matrixB(B);

        // Start recording time
        auto start = std::chrono::high_resolution_clock::now();

        // Solve A * X = B, storing the result in B
        trmm_blocked_mixed(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG,
                           real_t(1), A, B, W, TrmmBlockedOpts(nb));

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Use the minimum time
        mintime = std::min(double(elapsed.count()), mintime);
    }

    // Output
    std::cout << "gflops: " << flopTrmm / mintime << std::endl;

    // if (comparePerformance) {
    //     double mintime = 999999;  // in seconds
    //     for (int r = 0; r < nruns; ++r) {
    //         // Generate m-by-m upper-triangle random matrix
    //         gen_matrixA(A);

    //         // Generate m-by-n random matrix
    //         gen_matrixB(B);

    //         // Start recording time
    //         auto start = std::chrono::high_resolution_clock::now();

    //         // Solve A * X = B, storing the result in B
    //         trmm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG,
    //         real_t(1),
    //              A, B);

    //         // Record end time
    //         auto end = std::chrono::high_resolution_clock::now();

    //         // Compute elapsed time in nanoseconds
    //         auto elapsed =
    //         std::chrono::duration_cast<std::chrono::nanoseconds>(
    //             end - start);

    //         // Use the minimum time
    //         mintime = std::min(double(elapsed.count() * 1.0e-9), mintime);
    //     }

    //     std::cout << "Time TRMM in C++: " << mintime << " s" << std::endl;
    // }

#ifdef TLAPACK_USE_MKL
    if (comparePerformance && std::is_same_v<T1, float>) {
        float* Afp32;
        Afp32 = (float*)mkl_malloc(sizeof(float) * m * m, 64);

        double mintime = 999999 * 1e9;  // in nanoseconds
        for (int r = 0; r < nruns; ++r) {
            // Generate m-by-m upper-triangle random matrix
            gen_matrixA(A);

            // Generate m-by-n random matrix
            gen_matrixB(B);

            // Convert A to float
            for (int j = 0; j < m; ++j)
                for (int i = 0; i <= j; ++i)
                    Afp32[i + j * m] = (float)A(i, j);

            // Start recording time
            auto start = std::chrono::high_resolution_clock::now();

            // Solve A * X = B, storing the result in B
            cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                        CblasNonUnit, m, n, real_t(1), Afp32, m, B.ptr, B.ldim);

            // Record end time
            auto end = std::chrono::high_resolution_clock::now();

            // Compute elapsed time in nanoseconds
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start);

            // Use the minimum time
            mintime = std::min(double(elapsed.count()), mintime);
        }

        mkl_free(Afp32);

        std::cout << "gflops TRMM in MKL: " << flopTrmm / mintime << std::endl;
    }
#endif
}

int main()
{
    const int nruns = 5;
    const int m = 16384;
    const int n = 64;

    srand(3);

    const int L1cacheSize = 48 * 1024;
    const int L2cacheSize = 2048 * 1024;
    const int L3cacheSize = 107520 * 1024;

    // 2*nb*nb + 4*n*nb = LxcacheSize
    const int nb1 =
        min(int((sqrt((4. * n) * n + L1cacheSize * 2.) - 2 * n) / 2), m);
    const int nb2 =
        min(int((sqrt((4. * n) * n + L2cacheSize * 2.) - 2 * n) / 2), m);
    const int nb3 =
        min(int((sqrt((4. * n) * n + L3cacheSize * 2.) - 2 * n) / 2), m);

    // n = nb/2 in the L1 cache
    // 4*nb*nb = L1cacheSize
    const int nb4 = min(int(sqrt(L1cacheSize / 4.)), m);

    // Experiment 1 (measuring optimal block size)

    // std::set<int> nbset;
    // nbset.insert(nb1);
    // nbset.insert(nb2);
    // nbset.insert(nb3);
    // for (int i = 32; i < 512; i *= 2)
    //     nbset.insert(i);

    // const bool comparePerformance = false;
    // for (auto nb : nbset)
    //     run<std::pair<float, Eigen::bfloat16>>(nruns, m, n, nb,
    //                                            comparePerformance);

    // Experiment 2 (GFLOPS with fixed n=64)

    // const bool comparePerformance = true;
    // for (int m = 128; m <= 65536; m *= 2) {
    //     const int n = 64;
    //     run<std::pair<float, Eigen::bfloat16>>(nruns, m, n, nb4,
    //                                            comparePerformance);
    // }

    // Experiment 3 (GFLOPS with fixed n=128)

    const bool comparePerformance = true;
    for (int m = 128; m <= 32768; m *= 2) {
        const int n = 128;
        const int nb =
            min(int((sqrt((4. * n) * n + L1cacheSize * 2.) - 2 * n) / 2), m);
        run<std::pair<float, Eigen::bfloat16>>(nruns, m, n, nb4,
                                               comparePerformance);
    }

    // Experiment 4 (GFLOPS with fixed n=256)

    // const bool comparePerformance = true;
    // for (int m = 128; m <= 32768; m *= 2) {
    //     const int n = 256;
    //     run<std::pair<float, Eigen::bfloat16>>(nruns, m, n, nb4,
    //                                            comparePerformance);
    // }

    return 0;
}