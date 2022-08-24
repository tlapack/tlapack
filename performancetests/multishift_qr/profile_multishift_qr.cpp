/// @file example_gehd2.cpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tlapack/legacy_api/base/utils.hpp"
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack.hpp>

#include <memory>
#include <vector>
#include <chrono> // for high_resolution_clock
#include <iostream>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
inline void printMatrix(const matrix_t &A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i)
    {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

class rand_generator
{

private:
    const uint64_t a = 6364136223846793005;
    const uint64_t c = 1442695040888963407;
    uint64_t state = 1302;

public:
    uint32_t min()
    {
        return 0;
    }

    uint32_t max()
    {
        return UINT32_MAX;
    }

    void seed(uint64_t s)
    {
        state = s;
    }

    uint32_t operator()()
    {
        state = state * a + c;
        return state >> 32;
    }
};

template <typename T>
T rand_helper(rand_generator &gen)
{
    return static_cast<T>(gen()) / static_cast<T>(gen.max());
}
extern "C"
{
    void fortran_slaqr0(const bool &wantt, const bool &wantz, const int &n, const int &ilo, const int &ihi, float *H, const int &ldh, float *wr, float *wi, float *Z, const int &ldz, int &info, int &n_aed, int &n_sweep, int &n_shifts);
    void fortran_sgehrd(const int &n, const int &ilo, const int &ihi, float *H, const int &ldh, float *tau, int &info);
    void fortran_sorghr(const int &n, const int &ilo, const int &ihi, float *H, const int &ldh, float *tau, int &info);
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t n, int seed, bool use_fortran)
{
    srand(1); // Init random seed

    using real_t = tlapack::real_type<T>;
    using std::size_t;
    using matrix_t = tlapack::legacyMatrix<real_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    rand_generator gen;
    gen.seed(seed);

    // Arrays
    std::unique_ptr<T[]> A_(new T[lda * n]); // m-by-n
    std::unique_ptr<T[]> H_(new T[ldh * n]); // n-by-n
    std::unique_ptr<T[]> Q_(new T[ldq * n]); // m-by-n
    std::vector<T> tau(n);

    // Matrix views
    auto A = new_matrix(&A_[0], n, n);
    auto H = new_matrix(&H_[0], n, n);
    auto Q = new_matrix(&Q_[0], n, n);

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j)
    {
        for (size_t i = 0; i < n; ++i)
        {
            A(i, j) = static_cast<float>(0xDEADBEEF);
            Q(i, j) = static_cast<float>(0xCAFED00D);
        }
        for (size_t i = 0; i < n; ++i)
        {
            H(i, j) = static_cast<float>(0xFEE1DEAD);
        }
        tau[j] = static_cast<float>(0xFFBADD11);
    }

    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            A(i, j) = rand_helper<T>(gen);
    // for (size_t j = 0; j < n; ++j)
    //     for (size_t i = j + 2; i < n; ++i)
    //         A(i, j) = 0.0;

    // Frobenius norm of A
    auto normA = tlapack::lange(tlapack::frob_norm, A);

    // Copy A to Q
    tlapack::lacpy(tlapack::Uplo::General, A, Q);

    // 1) Compute A = QHQ* (Stored in the matrix Q)

    // Record start time
    auto startQHQ = std::chrono::high_resolution_clock::now();
    {
        int err;
        // Hessenberg factorization
        if(use_fortran){
            T *_tau = tau.data();
            fortran_sgehrd(n, 1, n, Q.ptr, Q.ldim, _tau, err);
        }else{
            err = tlapack::gehrd(0, n, Q, tau);
        }
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
        int err;
        // Generate Q = H_1 H_2 ... H_n
        // std::vector<T> work(n);
        // err = tlapack::unghr(0, n, Q, tau, work);
        T* _tau = tau.data();
        fortran_sorghr( n, 1, n, Q.ptr, n, _tau, err );
    }
    // Record end time
    auto endQ = std::chrono::high_resolution_clock::now();

    // Remove junk from lower half of H
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            H(i, j) = 0.0;

    // Record start time
    int n_aed = 0;
    int n_sweep = 0;
    int n_shifts_total = 0;
    auto startSchur = std::chrono::high_resolution_clock::now();
    {
        // Shur factorization
        int err;
        if (use_fortran)
        {
            std::unique_ptr<T[]> _wr(new T[n]);
            std::unique_ptr<T[]> _wi(new T[n]);
            fortran_slaqr0( true, true, n, 1, n, H.ptr, n, &_wr[0], &_wi[0], Q.ptr, n, err, n_aed, n_sweep, n_shifts_total );
        }
        else
        {
            tlapack::francis_opts_t<TLAPACK_SIZE_T, T> opts;
            std::vector<std::complex<real_t>> w(n);
            err = tlapack::multishift_qr(true, true, 0, n, H, w, Q, opts);
            // err = tlapack::lahqr(true, true, 0, n, H, w, Q);
            n_aed = opts.n_aed;
            n_sweep = opts.n_sweep;
            n_shifts_total = opts.n_shifts_total;
        }
    }
    // Record end time
    auto endSchur = std::chrono::high_resolution_clock::now();

    // Remove junk from lower half of H
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            H(i, j) = 0.0;

    // Compute elapsed time in nanoseconds
    auto elapsedQHQ = std::chrono::duration_cast<std::chrono::nanoseconds>(endQHQ - startQHQ);
    auto elapsedQ = std::chrono::duration_cast<std::chrono::nanoseconds>(endQ - startQ);
    auto elapsedSchur = std::chrono::duration_cast<std::chrono::nanoseconds>(endSchur - startSchur);

    real_t norm_orth_1, norm_repres_1;

    // 2) Compute ||Q'Q - I||_F

    {
        std::unique_ptr<T[]> _work(new T[n * n]);
        auto work = new_matrix(&_work[0], n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        tlapack::laset(tlapack::Uplo::General, (T)0.0, (T)1.0, work);
        // work receives Q'Q - I
        // tlapack::syrk( tlapack::Uplo::Upper, tlapack::Op::ConjTrans, (T) 1.0, Q, (T) -1.0, work );
        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans, (T)1.0, Q, Q, (T)-1.0, work);

        // Compute ||Q'Q - I||_F
        norm_orth_1 = tlapack::lansy(tlapack::frob_norm, tlapack::Uplo::Upper, work);
    }

    // 3) Compute ||QHQ* - A||_F / ||A||_F

    std::unique_ptr<T[]> H_copy_(new T[n * n]);
    auto H_copy = new_matrix(&H_copy_[0], n, n);
    tlapack::lacpy(tlapack::Uplo::General, H, H_copy);
    {
        std::unique_ptr<T[]> _work(new T[n * n]);
        auto work = new_matrix(&_work[0], n, n);

        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, (T)1.0, Q, H, work);
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans, (T)1.0, work, Q, H);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                H(i, j) -= A(i, j);

        norm_repres_1 = tlapack::lange(tlapack::frob_norm, H) / normA;
    }

    std::cout << std::endl;
    std::cout << "Hessenberg time = " << elapsedQHQ.count() * 1.0e-9 << " s" << std::endl;
    std::cout << "Q forming time = " << elapsedQ.count() * 1.0e-9 << " s" << std::endl;
    std::cout << "QR time = " << elapsedSchur.count() * 1.0e-9 << " s" << std::endl;
    std::cout << "||QHQ* - A||_F/||A||_F  = " << norm_repres_1
              << ",        ||Q'Q - I||_F  = " << norm_orth_1 << std::endl;
    std::cout << "AED calls   " << n_aed << std::endl
              << "Sweep calls " << n_sweep << std::endl
              << "Shifts used " << n_shifts_total << std::endl;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int n, use_lapack, seed;

    // Default arguments
    n = (argc < 2) ? 100 : atoi(argv[1]);
    seed =  (argc < 3) ? 1302 : atoi(argv[2]);
    use_lapack = (argc < 4) ? -1 : atoi(argv[3]);

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    if (use_lapack == -1 or use_lapack == 0)
    {
        printf("run< float >( %d ) without LAPACK \n", n);
        run<float>(n, seed, false);
        printf("-----------------------\n");
    }

    if (use_lapack == -1 or use_lapack == 1)
    {
        printf("run< float >( %d ) with LAPACK \n", n);
        run<float>(n, seed, true);
        printf("-----------------------\n");
    }
    
    return 0;
}
