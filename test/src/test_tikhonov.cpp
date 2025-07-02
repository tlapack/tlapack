#include "testutils.hpp"
//
#include <tlapack/lapack/tik_bidiag_elden.hpp>
#include <tlapack/lapack/tik_qr.hpp>
#include <tlapack/lapack/tik_svd.hpp>
#include <tlapack/lapack/tkhnv.hpp>
#include <tlapack/plugins/stdvector.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Testing all cases of Tikhonov",
                   "[tikhonov check]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    Create<matrix_t> new_matrix;

    const idx_t m = GENERATE(1, 2, 12, 20, 30);
    const idx_t n = GENERATE(1, 2, 3, 7, 8);
    const idx_t k = GENERATE(1, 7, 12, 19);
    const real_t lambda = GENERATE(1e-6, 7.5, 1e6);

    using variant_t = TikVariant;
    const variant_t variant =
        GENERATE((variant_t(TikVariant::QR)), (variant_t(TikVariant::Elden)),
                 (variant_t(TikVariant::SVD)));
    DYNAMIC_SECTION(" m = " << m << " n = " << n << " k = " << k << " lambda = "
                            << lambda << " variant = " << (char)variant)

    {
        if (m >= n) {
            // eps is the machine precision, and tol is the tolerance we accept
            // for tests to pass

            real_t tol;
            const real_t eps = ulp<real_t>();
            if constexpr (is_complex<T>)
                tol = 1 * real_t(max(m, k)) * eps;
            else
                tol = 1 * real_t(max(m, k)) * eps;

            // Declare matrices
            std::vector<T> A_;
            auto A = new_matrix(A_, m, n);
            std::vector<T> A_copy_;
            auto A_copy = new_matrix(A_copy_, m, n);
            std::vector<T> b_;
            auto b = new_matrix(b_, m, k);
            std::vector<T> bcopy_;
            auto bcopy = new_matrix(bcopy_, m, k);
            std::vector<T> x_;
            auto x = new_matrix(x_, n, k);
            std::vector<T> y_;
            auto y = new_matrix(y_, n, k);

            // Initializing matrices randomly            std::cout << "\nnormr =
            // " << normr;

            // tik_qr(A, b, lambda);
            // tik_bidiag_elden(A, b, lambda);
            // tik_qr failed all 8 test cases
            // tik_svd(A, b, lambda);

            TikOpts opts;
            opts.variant = variant;
            tkhnv(A, b, lambda, opts);

            // Check routine
            lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), x);

            // Compute b - A *x -> b
            gemm(NO_TRANS, NO_TRANS, real_t(-1), A_copy, x, real_t(1), bcopy);

            // Compute A.H*(b - A x) -> y
            gemm(CONJ_TRANS, NO_TRANS, real_t(1), A_copy, bcopy, y);

            // Compute A.H*(b - A x) - (lambda^2)*x -> y
            for (idx_t j = 0; j < k; j++)
                for (idx_t i = 0; i < n; i++)
                    y(i, j) -= (lambda) * (lambda)*x(i, j);

            real_t normr = lange(FROB_NORM, y);

            real_t normA = lange(FROB_NORM, A_copy);

            real_t normb = lange(FROB_NORM, bcopy);

            real_t normx = lange(FROB_NORM, x);

            if (normr > tol * (normA * (normb + normA * normx) +
                               abs(lambda) * abs(lambda) * normx)) {
                std::cout << "\nnormr = " << normr;

                std::cout << "\ntol * scalar= "
                          << tol * (normA * (normb + normA * normx) +
                                    abs(lambda) * abs(lambda) * normx)
                          << "\n";
            }
            CHECK(normr <= tol * (normA * (normb + normA * normx) +
                                  abs(lambda) * abs(lambda) * normx));
        }
    }
}