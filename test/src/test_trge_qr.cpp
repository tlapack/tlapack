#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/trge_qr2.hpp>
#include <tlapack/lapack/trge_ung2r.hpp>
#include <tlapack/lapack/ung2r.hpp>

#include "testutils.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE("Triangle on top of general matrix QR factorization",
                   "[trge_qr2][trge_ung2r][trge_unm2r]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;

    using range = pair<idx_t, idx_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    // Not sure if I should inlude the 1x1 case and k = 1
    // idx_t m = GENERATE(1, 5, 11, 23);
    // idx_t n = GENERATE(1, 7, 16, 22);
    // idx_t k = GENERATE(1, 4, 14, 21);

    idx_t m = GENERATE(5, 11, 23);
    idx_t n = GENERATE(7, 16, 22);
    idx_t k = GENERATE(4, 14, 21);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " k = " << k)
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        std::vector<T> A_;
        auto A = new_matrix(A_, n + m, n);
        std::vector<T> R_;
        auto R = new_matrix(R_, n, n);
        std::vector<T> Q_;
        auto Q = new_matrix(Q_, n + m, k);

        std::vector<T> tau(n);

        // Randomly initialize matrices
        MatrixMarket mm;

        mm.random(A);
        mm.random(Q);
        mm.random(R);

        auto Q1 = slice(Q, range{0, m + n}, range{0, n});
        lacpy(GENERAL, A, Q1);

        trge_qr2(Q1, tau);

        lacpy(UPPER_TRIANGLE, slice(Q1, range{0, n}, range{0, n}), R);

        trge_ung2r(Q, tau);

        // Put zeros below Upper Triangle R in A
        for (idx_t j = 0; j < n - 1; j++)
            for (idx_t i = j + 1; i < n; i++)
                Q(i, j) = real_t(0);

        // 2) Compute ||Qᴴ Q - I||_F

        std::vector<T> work_;
        auto work = new_matrix(work_, k, k);

        // work receives the identity n*n
        laset(UPPER_TRIANGLE, real_t(0.0), real_t(1.0), work);

        // work receives QᴴQ - I
        herk(UPPER_TRIANGLE, CONJ_TRANS, real_t(1.0), Q, real_t(-1.0), work);

        // Compute ||QᴴQ - I||_F
        real_t norm_orth = lanhe(FROB_NORM, UPPER_TRIANGLE, work);

        {
            // std::vector<T> work_;
            // auto work = new_matrix(work_, k, k);

            // // work receives the identity n*n
            // laset(UPPER_TRIANGLE, real_t(0.0), real_t(1.0), work);

            // // work receives QᴴQ - I
            // herk(UPPER_TRIANGLE, CONJ_TRANS, real_t(1.0), Q, real_t(-1.0),
            //      work);

            // // Compute ||QᴴQ - I||_F
            // real_t norm_orth = lanhe(FROB_NORM, UPPER_TRIANGLE, work);
        }

        // 3) Compute ||QR - A||_F / ||A||_F

        for (idx_t j = 0; j < n; j++)
            for (idx_t i = j + 1; i < n; i++)
                A(i, j) = real_t(0);

        // // Frobenius norm of A
        auto normA = lange(FROB_NORM, A);

        // Copy Q1 to work
        lacpy(GENERAL, Q1, work);

        trmm(RIGHT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1.0),
             R, work);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m + n; ++i)
                work(i, j) -= A(i, j);

        real_t norm_repres = lange(FROB_NORM, work) / normA;

        {
            // for (idx_t j = 0; j < n; j++)
            //     for (idx_t i = j + 1; i < n; i++)
            //         A(i, j) = real_t(0);

            // // // Frobenius norm of A
            // auto normA = lange(FROB_NORM, A);

            // std::vector<T> work_;
            // auto work = new_matrix(work_, m + n, n);
            // for (size_t j = 0; j < n; ++j)
            //     for (size_t i = 0; i < m + n; ++i)
            //         work(i, j) = static_cast<float>(0xABADBABE);

            // // Copy Q1 to work
            // lacpy(GENERAL, Q1, work);

            // trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
            //      real_t(1.0), R, work);
            // std::cout << std::endl;

            // for (size_t j = 0; j < n; ++j)
            //     for (size_t i = 0; i < m + n; ++i)
            //         work(i, j) -= A(i, j);

            // real_t norm_repres = lange(FROB_NORM, work) / normA;
        }
    }
}