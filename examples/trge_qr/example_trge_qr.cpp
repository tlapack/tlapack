#include <tlapack/plugins/legacyArray.hpp>
//
#include <../../tlapack/test/include/MatrixMarket.hpp>

// <T>LAPACK
#include <tlapack/blas/herk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/larf.hpp>
#include <tlapack/lapack/larfg.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/trge_qr2.hpp>
#include <tlapack/lapack/trge_ung2r.hpp>
#include <tlapack/lapack/ung2r.hpp>

using namespace tlapack;

template <typename T>
void run(size_t m, size_t n, size_t k)
{
    // Create utilities for code
    using matrix_t = LegacyMatrix<T>;
    using real_t = real_type<T>;

    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    // Declare Arrays
    std::vector<T> tau(n);

    // Declare Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, n + m, n);
    std::vector<T> A0_;
    auto A0 = new_matrix(A0_, n + m, n);
    std::vector<T> A1_;
    auto A1 = new_matrix(A1_, n + m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, n + m, k);

    // std::vector<T> Q0_;
    // auto Q0 = new_matrix(Q0_, n, k);
    // std::vector<T> Q1_;
    // auto Q1 = new_matrix(Q1_, m, k);

    // Randomly initialize matrices
    MatrixMarket mm;

    mm.random(A);
    mm.random(Q);
    mm.random(R);

    auto Qthin = slice(Q, range{0, m + n}, range{0, n});
    lacpy(GENERAL, A, Qthin);

    auto Qthin0 = slice(Qthin, range{0, n}, range{0, n});
    auto Qthin1 = slice(Qthin, range{n, n + m}, range{0, n});

    trge_qr2(Qthin0, Qthin1, tau);

    lacpy(UPPER_TRIANGLE, slice(Qthin, range{0, n}, range{0, n}), R);

    ///////////////////////////////////////////////////////////////////////////////

    auto Q0 = slice(Q, range{0, n}, range{0, k});
    auto Q1 = slice(Q, range{n, n + m}, range{0, k});

    // trge_ung2r(Q, tau);
    trge_ung2r(Q0, Q1, tau);

    // Put zeros below Upper Triangle R in A
    for (idx_t j = 0; j < n - 1; j++)
        for (idx_t i = j + 1; i < n; i++)
            Q(i, j) = real_t(0);

    real_t norm_orth, norm_repres;

    // 2) Compute ||Qᴴ Q - I||_F

    {
        std::vector<T> work_;
        auto work = new_matrix(work_, k, k);
        for (size_t j = 0; j < k; ++j)
            for (size_t i = 0; i < k; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        laset(UPPER_TRIANGLE, 0.0, 1.0, work);

        // work receives QᴴQ - I
        herk(Uplo::Upper, Op::ConjTrans, real_t(1.0), Q, real_t(-1.0), work);

        // Compute ||QᴴQ - I||_F
        norm_orth = lanhe(FROB_NORM, UPPER_TRIANGLE, work);
    }

    // 3) Compute ||QR - A||_F / ||A||_F

    {
        for (idx_t j = 0; j < n; j++)
            for (idx_t i = j + 1; i < n; i++)
                A(i, j) = real_t(0);

        // // Frobenius norm of A
        auto normA = lange(FROB_NORM, A);

        std::vector<T> work_;
        auto work = new_matrix(work_, m + n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m + n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // Copy Qthin to work
        lacpy(GENERAL, Qthin, work);

        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, real_t(1.0),
             R, work);
        std::cout << std::endl;

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m + n; ++i)
                work(i, j) -= A(i, j);

        norm_repres = lange(FROB_NORM, work) / normA;
    }

    // // *) Output

    std::cout << std::endl;
    std::cout << "||QR - A||_F/||A||_F  = " << norm_repres
              << ",        ||Qᴴ Q - I||_F  = " << norm_orth;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n, k;

    // Default arguments
    m = (argc < 2) ? 3 : atoi(argv[1]);
    n = (argc < 3) ? 2 : atoi(argv[2]);
    k = (argc < 4) ? 2 : atoi(argv[3]);

    // k is between n and m+n

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< complex<double> >( %d, %d, %d )", m, n, k);
    run<std::complex<double> >(m, n, k);
    std::cout << "-----------------------" << std::endl;

    return 0;
}
