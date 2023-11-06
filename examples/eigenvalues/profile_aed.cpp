/// @file example_gehd2.cpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

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
void fortran_slaqr2(const bool* wantt,
                    const bool* wantz,
                    const int* n,
                    const int* ilo,
                    const int* ihi,
                    const int* nw,
                    float* H,
                    const int* ldh,
                    float* Z,
                    const int* ldz,
                    int* ns,
                    int* nd,
                    float* sr,
                    float* si);
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t n, size_t nw, bool use_fortran)
{
    using real_t = tlapack::real_type<T>;
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;

    constexpr bool use_lapack
#ifdef USE_LAPACK
        = std::is_same_v<T, float>;
#else
        = false;
#endif

    // Functor for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    rand_generator gen;
    gen.seed(1302);

    // Constants
    int one = 1, len = n;
    bool yes = true;

    // Vectors
    std::vector<T> tau(n);

    // Matrices
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
        for (size_t i = 0; i < std::min(j + 2, n); ++i)
            A(i, j) = rand_helper<T>(gen);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            A(i, j) = 0.0;

    // Frobenius norm of A
    auto normA = tlapack::lange(tlapack::FROB_NORM, A);

    // Copy A to H
    tlapack::lacpy(tlapack::Uplo::General, A, H);
    tlapack::laset(tlapack::Uplo::General, (T)0.0, (T)1.0, Q);

    size_t ls, ld;
    // Record start time
    auto startTime = std::chrono::high_resolution_clock::now();
    {
        // Shur factorization
        if constexpr (use_lapack) {
            if (use_fortran) {
                std::vector<T> wr(n);
                std::vector<T> wi(n);
                int ls2, ld2, nwInt = nw;
                fortran_slaqr2(&yes, &yes, &len, &one, &len, &nwInt, H.ptr,
                               &len, Q.ptr, &len, &ls2, &ld2, wr.data(),
                               wi.data());
                ls = ls2;
                ld = ld2;
            }
            else {
                std::vector<std::complex<real_t>> w(n);
                tlapack::aggressive_early_deflation(true, true, 0, n, nw, H, w,
                                                    Q, ls, ld);
            }
        }
        else {
            std::vector<std::complex<real_t>> w(n);
            tlapack::aggressive_early_deflation(true, true, 0, n, nw, H, w, Q,
                                                ls, ld);
        }
    }
    // Record end time
    auto endTime = std::chrono::high_resolution_clock::now();

    // Remove junk from lower half of H
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            H(i, j) = 0.0;

    // Compute elapsed time in nanoseconds
    auto elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
        endTime - startTime);

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
            tlapack::lansy(tlapack::FROB_NORM, tlapack::Uplo::Upper, work);
    }

    // 3) Compute ||QHQ* - A||_F / ||A||_F

    std::vector<T> H_copy_;
    auto H_copy = new_matrix(H_copy_, n, n);
    tlapack::lacpy(tlapack::Uplo::General, H, H_copy);
    {
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);

        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, (T)1.0, Q, H,
                      work);
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans, (T)1.0,
                      work, Q, H);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                H(i, j) -= A(i, j);

        norm_repres_1 = tlapack::lange(tlapack::FROB_NORM, H) / normA;
    }

    std::cout << std::endl;
    std::cout << "AED time = " << elapsedTime.count() * 1.0e-9 << " s"
              << std::endl;
    std::cout << "||QHQ* - A||_F/||A||_F  = " << norm_repres_1
              << ",        ||Q'Q - I||_F  = " << norm_orth_1 << std::endl;
    std::cout << "Number of shifts   " << ls << std::endl
              << "Deflations " << ld << std::endl;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int n, nw;
    bool use_lapack;

    // Default arguments
    n = (argc <= 1) ? 100 : atoi(argv[1]);
    nw = (argc <= 2) ? 20 : atoi(argv[2]);
    use_lapack = (argc <= 3) ? false : atoi(argv[3]);

    // Usage
    if (argc > 4) {
        std::cout << "Usage: " << argv[0] << " [n] [nw] [use_lapack]"
                  << std::endl;
        std::cout << "n: matrix size" << std::endl;
        std::cout << "nw: Desired window size to perform aggressive early "
                     "deflation on"
                  << std::endl;
        std::cout << "use_lapack: 1 (true) or 0 (false), use lapack or not"
                  << std::endl;
        return 1;
    }

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float >( %d )", n);
    run<float>(n, nw, use_lapack);
    printf("-----------------------\n");

    printf("run< double >( %d )", n);
    run<double>(n, nw, use_lapack);
    printf("-----------------------\n");

    printf("run< long double >( %d )", n);
    run<long double>(n, nw, use_lapack);
    printf("-----------------------\n");

    return 0;
}
