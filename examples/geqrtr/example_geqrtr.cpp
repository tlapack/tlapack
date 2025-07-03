#include <tlapack/plugins/legacyArray.hpp>
//
#include <../../tlapack/test/include/MatrixMarket.hpp>

// <T>LAPACK
#include <tlapack/blas/herk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/larf.hpp>
#include <tlapack/lapack/larfg.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

using namespace tlapack;
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

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t k)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using real_t = real_type<T>;

    using idx_t = tlapack::size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = true;

    // Arrays
    std::vector<T> tau(n);

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, n + m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m + n, k);

    // auto A0 = slice(A, range{0, n}, range{0, n});
    // auto A1 = slice(A, range{n, m + n}, range{0, n});

    MatrixMarket mm;

    mm.random(A);
    mm.random(Q);
    mm.random(R);

    for (idx_t j = 0; j < n; j++)
        for (idx_t i = 0; i < m + n; i++)
            Q(i, j) = real_t(0xDEADBEEF);

    for (idx_t j = 0; j < n; j++)
        for (idx_t i = j + 1; i < n; i++)
            A(i, j) = real_t(0xDEADBEEF);

    std::cout << "\nA before geqrf = \n";
    printMatrix(A);

    // tlapack::lacpy(tlapack::GENERAL, A, Q);
    // geqr2(Q, tau);

    std::vector<T> Acpy_;
    auto Acpy = new_matrix(Acpy_, n + m, n);
    tlapack::lacpy(tlapack::GENERAL, A, Acpy);
    // geqr2(A, tau);

    ///////////////////// Begin Algorithm ////////////////////////////

    // std::vector<T> work_;
    // auto work = new_matrix(work_, m + n, 1);

    // for (idx_t i = 0; i < n - 1; ++i) {
    //     // auto v = slice(A, range{i, m + n}, i);
    //     // larfg(FORWARD, COLUMNWISE_STORAGE, v, tau[i]);

    //     // auto v1 = slice(A, range{i + 1, m + n}, i);

    //     // THIS IS MY CHANGE
    //     auto v1 = slice(A, range{n - 1, m + n}, i);

    //     // generates elem HH reflector
    //     larfg(COLUMNWISE_STORAGE, A(i, i), v1, tau[i]);

    //     // Define C := A[i:m,i+1:n]

    //     // auto C = slice(A, range{i, m + n}, range{i + 1, n});
    //     // v = slice(A, range{i, m + n}, i);
    //     // C := ( I - conj(tau_i) v v^H ) C
    //     // larf_work(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE, v, conj(tau[i]),
    //     C,
    //     //           work);

    //     auto v2 = slice(A, range{i + 1, m + n}, i);
    //     auto C0 = slice(A, i, range{i + 1, n});  // THIS IS GOOD

    //     auto C1 = slice(A, range{i + 1, m + n}, range{i + 1, n});
    //     larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v2, conj(tau[i]), C0, C1,
    //               work);
    // }
    // // Define v : = A [n - 1:m, n - 1]
    // auto v3 = slice(A, range{n, m + n}, n - 1);
    // // Generate the n-th elementary Householder reflection on v
    // larfg(COLUMNWISE_STORAGE, A(n - 1, n - 1), v3, tau[n - 1]);

    // tlapack::lacpy(tlapack::GENERAL, A, Q);
    // tlapack::lacpy(tlapack::GENERAL, Acpy, A);

    // std::cout << "\nA after geqrf = \n";
    // printMatrix(Q);

    // // // Save the R matrix
    // tlapack::lacpy(tlapack::UPPER_TRIANGLE, Q, R);

    // // // Generates Q = H_1 H_2 ... H_n
    // tlapack::ung2r(Q, tau);

    // std::cout << "\nA after geqrf = \n";
    // printMatrix(Q);

    std::vector<T> work_;
    auto work = new_matrix(work_, m + n, 1);

    for (idx_t i = 0; i < n - 1; ++i) {
        auto v1 = slice(A, range{n, m + n}, i);
        larfg(COLUMNWISE_STORAGE, A(i, i), v1, tau[i]);

        auto v2 = slice(A, range{n, m + n}, i);
        auto C0 = slice(A, i, range{i + 1, n});
        auto C1 = slice(A, range{n, m + n}, range{i + 1, n});
        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v2, conj(tau[i]), C0, C1,
                  work);
    }
    // Define v : = A [n - 1:m, n - 1]
    auto v3 = slice(A, range{n, m + n}, n - 1);
    // Generate the n-th elementary Householder reflection on v
    larfg(COLUMNWISE_STORAGE, A(n - 1, n - 1), v3, tau[n - 1]);

    for (idx_t j = 0; j < n; j++)
        for (idx_t i = j + 1; i < n; i++)
            A(i, j) = real_t(0);

    auto Q1 = slice(Q, range{0, m + n}, range{0, n});

    tlapack::lacpy(tlapack::GENERAL, A, Q1);

    // std::cout << "\nA after geqrf = \n";
    // printMatrix(Q);

    // // Save the R matrix
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, Q1, R);

    // // Generates Q = H_1 H_2 ... H_n
    tlapack::ung2r(Q, tau);

    std::vector<T> W_;
    auto W = new_matrix(W_, m + n, k);

    for (idx_t j = 0; j < n; j++)
        for (idx_t i = 0; i < m + n; i++)
            W(i, j) = real_t(0xDEADBEEF);

    tlapack::laset(tlapack::GENERAL, real_t(0.0), real_t(1.0), W);

    auto v = slice(A, range{n, m + n}, n - 1);
    auto W0 = slice(W, n - 1, range{n, k});
    auto W1 = slice(W, range{n, m + n}, range{n, k});
    larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v, tau[n - 1], W0, W1, work);

    // W0 = slice(W, n - 1, range{n - 1, n});
    // W1 = slice(W, range{n, m + n}, range{n - 1, n});
    // larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v, tau[n - 1], W0, W1, work);

    scal(-tau[n - 1], v);
    T one = T(1);
    W(n - 1, n - 1) = one - tau[n - 1];
    for (idx_t j = 0; j < m; j++)
        W(n + j, n - 1) = v[j];

    for (idx_t i = n - 1; i-- > 0;) {
        auto v = slice(A, range{n, m + n}, i);
        auto W0 = slice(W, i, range{i + 1, k});
        auto W1 = slice(W, range{n, m + n}, range{i + 1, k});
        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v, tau[i], W0, W1, work);

        // W0 = slice(W, i, range{i, i + 1});
        // W1 = slice(W, range{n, m + n}, range{i, i + 1});
        // larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v, tau[i], W0, W1, work);
    }
    // std::cout << "\nA after geqrf = \n";
    // printMatrix(Q);

    lacpy(GENERAL, W, Q);

    ///////////////// I THINK ALG ENDS HERE //////////////////////////
    tlapack::lacpy(tlapack::GENERAL, Acpy, A);

    real_t norm_orth, norm_repres;

    // 2) Compute ||Qᴴ Q - I||_F

    {
        std::vector<T> work_;
        auto work = new_matrix(work_, k, k);
        for (size_t j = 0; j < k; ++j)
            for (size_t i = 0; i < k; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        tlapack::laset(tlapack::UPPER_TRIANGLE, 0.0, 1.0, work);

        std::cout << "\nI =\n";
        printMatrix(work);

        // work receives QᴴQ - I
        tlapack::herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans, real_t(1.0),
                      Q, real_t(-1.0), work);

        // Compute ||QᴴQ - I||_F
        norm_orth =
            tlapack::lanhe(tlapack::FROB_NORM, tlapack::UPPER_TRIANGLE, work);
    }

    // 3) Compute ||QR - A||_F / ||A||_F

    {
        for (idx_t j = 0; j < n; j++)
            for (idx_t i = j + 1; i < n; i++)
                A(i, j) = real_t(0);

        // // Frobenius norm of A
        auto normA = tlapack::lange(tlapack::FROB_NORM, A);

        std::vector<T> work_;
        auto work = new_matrix(work_, m + n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m + n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // Copy Q to work
        tlapack::lacpy(tlapack::GENERAL, Q1, work);

        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper,
                      tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(1.0),
                      R, work);
        std::cout << std::endl;

        // printMatrix(work);
        // std::cout << std::endl;

        // printMatrix(A);
        // std::cout << std::endl;

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m + n; ++i)
                work(i, j) -= A(i, j);

        norm_repres = tlapack::lange(tlapack::FROB_NORM, work) / normA;

        printMatrix(work);
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
    k = (argc < 4) ? 5 : atoi(argv[3]);

    // k is between n and m+n

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< complex<double> >( %d, %d, %d )", m, n, k);
    run<std::complex<double> >(m, n, k);
    std::cout << "-----------------------" << std::endl;

    return 0;
}
