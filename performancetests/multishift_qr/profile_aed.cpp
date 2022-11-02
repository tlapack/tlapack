/// @file example_gehd2.cpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tlapack/plugins/legacyArray.hpp>
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
    void fortran_slaqr2(const bool &wantt, const bool &wantz, const int &n, const int &ilo, const int &ihi,
    const int &nw, float *H, const int &ldh, float *Z, const int &ldz, int &ns, int &nd, float *sr, float *si);

}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t n, size_t nw, bool use_fortran)
{
    srand(1); // Init random seed

    using real_t = tlapack::real_type<T>;
    using std::size_t;
    using matrix_t = tlapack::legacyMatrix<real_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    rand_generator gen;
    gen.seed(1302);

    // Vectors
    std::vector<T> tau(n);

    // Matrices
    std::vector<T> A_; auto A = new_matrix(A_, n, n);
    std::vector<T> H_; auto H = new_matrix(H_, n, n);
    std::vector<T> Q_; auto Q = new_matrix(Q_, n, n);

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
        for (size_t i = 0; i < j + 2; ++i)
            A(i, j) = rand_helper<T>(gen);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            A(i, j) = 0.0;

    // Frobenius norm of A
    auto normA = tlapack::lange(tlapack::frob_norm, A);

    // Copy A to H
    tlapack::lacpy(tlapack::Uplo::General, A, H);
    tlapack::laset(tlapack::Uplo::General, (T)0.0, (T)1.0, Q );

    size_t ls, ld;
    // Record start time
    auto startTime = std::chrono::high_resolution_clock::now();
    {
        // Shur factorization
        if (use_fortran)
        {
            std::unique_ptr<T[]> _wr(new T[n]);
            std::unique_ptr<T[]> _wi(new T[n]);
            int ls2, ld2;
            fortran_slaqr2(true, true, n, 1, n, nw, H.ptr, H.ldim, Q.ptr, n, ls2, ld2, &_wr[0], &_wi[0]);
            ls = ls2;
            ld = ld2;
        }
        else
        {
            std::vector<std::complex<real_t>> w(n);
            tlapack::agressive_early_deflation( true, true, (size_t)0, n, nw, H, w, Q, ls, ld );
        }
    }
    // Record end time
    auto endTime = std::chrono::high_resolution_clock::now();

    // Remove junk from lower half of H
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            H(i, j) = 0.0;

    // Compute elapsed time in nanoseconds
    auto elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);

    real_t norm_orth_1, norm_repres_1;

    // 2) Compute ||Q'Q - I||_F

    {
        std::vector<T> work_; auto work = new_matrix(work_, n, n);
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

    std::vector<T> H_copy_; auto H_copy = new_matrix(H_copy_, n, n);
    tlapack::lacpy(tlapack::Uplo::General, H, H_copy);
    {
        std::vector<T> work_; auto work = new_matrix(work_, n, n);

        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, (T)1.0, Q, H, work);
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans, (T)1.0, work, Q, H);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                H(i, j) -= A(i, j);

        norm_repres_1 = tlapack::lange(tlapack::frob_norm, H) / normA;
    }

    std::cout << std::endl;
    std::cout << "AED time = " << elapsedTime.count() * 1.0e-9 << " s" << std::endl;
    std::cout << "||QHQ* - A||_F/||A||_F  = " << norm_repres_1
              << ",        ||Q'Q - I||_F  = " << norm_orth_1 << std::endl;
    std::cout << "Number of shifts   " << ls << std::endl
              << "Deflations " << ld << std::endl;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int n, nw, use_lapack;

    // Default arguments
    n = (argc < 2) ? 100 : atoi(argv[1]);
    nw = (argc < 3) ? 20 : atoi(argv[2]);
    use_lapack = (argc < 4) ? -1 : atoi(argv[3]);

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    if (use_lapack == -1 or use_lapack == 0)
    {
        printf("run< float >( %d ) without LAPACK \n", n);
        run<float>(n, nw, false);
        printf("-----------------------\n");
    }

    if (use_lapack == -1 or use_lapack == 1)
    {
        printf("run< float >( %d ) with LAPACK \n", n);
        run<float>(n, nw, true);
        printf("-----------------------\n");
    }

    return 0;
}
