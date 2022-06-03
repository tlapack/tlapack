/// @file example_gehd2.cpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "legacy_api/base/utils.hpp"
#include <plugins/tlapack_stdvector.hpp>
#include <tlapack.hpp>

#include <memory>
#include <vector>
#include <chrono> // for high_resolution_clock
#include <iostream>

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

//------------------------------------------------------------------------------
template <typename T>
void run(size_t n, size_t ns, int seed, bool use_recursion)
{
    srand(1); // Init random seed

    using real_t = tlapack::real_type<T>;
    using std::size_t;
    using tlapack::internal::colmajor_matrix;

    rand_generator gen;
    gen.seed(seed);

    // Arrays
    std::unique_ptr<T[]> _A(new T[n * n]);
    std::unique_ptr<T[]> _Q(new T[n * n]);
    std::unique_ptr<T[]> _V(new T[3 * ns / 2]);

    // Matrix views
    auto A = colmajor_matrix<T>(&_A[0], n, n, n);
    auto Q = colmajor_matrix<T>(&_Q[0], n, n, n);
    auto V = colmajor_matrix<T>(&_V[0], 3, ns / 2, 3);
    std::vector<std::complex<real_t>> s(ns);

    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < std::min(j + 2, n); ++i)
            A(i, j) = rand_helper<T>(gen);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j + 2; i < n; ++i)
            A(i, j) = 0.0;

    // Pick some shifts to use
    for (size_t i = 0; i < ns; i = i + 2)
    {
        s[i] = std::complex<real_t>((real_t)i, (real_t) + 1);
        s[i + 1] = std::complex<real_t>((real_t)i, (real_t)-1);
    }

    // Frobenius norm of A
    // auto normA = tlapack::lange(tlapack::frob_norm, A);

    tlapack::introduce_bulges(A, s, Q, V);

    tlapack::move_bulges_opts_t<TLAPACK_SIZE_T, T> opts;
    opts.nx = 16;
    std::unique_ptr<T[]> _work(new T[2*n*n]);
    opts._work = &_work[0];
    opts.lwork = 2*n*n;

    // Record start time
    auto startTime = std::chrono::high_resolution_clock::now();
    {
        if (use_recursion)
        {
            tlapack::move_bulges_recursive(A, s, Q, V, opts);
        }
        else
        {
            tlapack::move_bulges(A, s, Q, V);
        }
    }
    // Record end time
    auto endTime = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);

    std::cout << std::endl;
    std::cout << "time = " << elapsed.count() * 1.0e-9 << " s" << std::endl;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int n, ns, use_recursion, seed;

    // Default arguments
    n = (argc < 2) ? 64 : atoi(argv[1]);
    ns = (argc < 3) ? 32 : atoi(argv[2]);
    seed = (argc < 4) ? 1302 : atoi(argv[3]);
    use_recursion = (argc < 5) ? -1 : atoi(argv[4]);

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    if (use_recursion == -1 or use_recursion == 0)
    {
        printf("run< float >( %d ) without recursion \n", n);
        run<float>(n, ns, seed, false);
        printf("-----------------------\n");
    }

    if (use_recursion == -1 or use_recursion == 1)
    {
        printf("run< float >( %d ) with recursion \n", n);
        run<float>(n, ns, seed, true);
        printf("-----------------------\n");
    }

    return 0;
}