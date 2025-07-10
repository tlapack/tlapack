#include <tlapack/plugins/legacyArray.hpp>
//
#include <tlapack/lapack/bidiag.hpp>
#include <tlapack/lapack/elden_elim.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/unmlq.hpp>
#include <tlapack/lapack/unmqr.hpp>

#include "../../test/include/MatrixMarket.hpp"
// Check function created for the example
#include "tik_check.hpp"

using namespace tlapack;

//------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t k)
{
    using real_t = real_type<T>;
    using matrix_t = LegacyMatrix<T>;
    using idx_t = size_type<matrix_t>;

    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    // Declare Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> b_;
    auto b = new_matrix(b_, m, k);
    std::vector<T> bcopy_;
    auto bcopy = new_matrix(bcopy_, m, k);
    std::vector<T> work_;
    auto work = new_matrix(work_, n, k);

    std::vector<T> x_;
    auto x = new_matrix(x_, n, k);
    auto x2 = slice(x, range{1, n}, range{0, k});

    real_t lambda;

    // Initializing matrices randomly
    MatrixMarket mm;
    mm.random(A);
    mm.random(b);

    // Create copies for check
    lacpy(GENERAL, A, A_copy);
    lacpy(GENERAL, b, bcopy);

    // Declare vectors
    std::vector<T> tauv(n);
    std::vector<T> tauw(n);
    std::vector<real_t> d(n);
    std::vector<real_t> e(n - 1);

    // Bidiagonal decomposition
    bidiag(A, tauv, tauw);

    // Apply to b

    unmqr(LEFT_SIDE, CONJ_TRANS, A, tauv, b);

    real_t normr_inside0 = lange(FROB_NORM, slice(b, range{n, m}, range{0, k}));
    real_t normr_outside;
    real_t check_tikhonov;

    real_t normx_inside;

    real_t normr_inside1;

    real_t normr_inside;
    real_t normx_outside;

    ///////////////////////////////////////////////////////////////////////////////

    lambda = 1e-2;

    lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), x);

    for (idx_t j = 0; j < n; ++j)
        d[j] = real(A(j, j));
    for (idx_t j = 0; j < n - 1; ++j)
        e[j] = real(A(j, j + 1));

    elden_elim(lambda, d, e, work, x);

    normx_inside = lange(FROB_NORM, x);

    normr_inside1 = lange(FROB_NORM, work);

    normr_inside =
        sqrt(normr_inside0 * normr_inside0 + normr_inside1 * normr_inside1 -
             lambda * lambda * normx_inside * normx_inside);

    unmlq(LEFT_SIDE, CONJ_TRANS, slice(A, range{0, n - 1}, range{1, n}),
          slice(tauw, range{0, n - 1}), x2);

    tik_check(A_copy, bcopy, lambda, x, &normr_outside, &check_tikhonov);

    normx_outside = lange(FROB_NORM, x);

    std::cout << std::endl;

    std::cout << "lambda = " << lambda << std::endl;

    std::cout << "check_tikhonov = " << check_tikhonov << std::endl;

    std::cout << "normr = ||b-Ax||_F  == (outside) " << normr_outside
              << " == (inside) " << normr_inside << " == (diff) "
              << abs(normr_outside - normr_inside) << std::endl;

    std::cout << "normx = ||x||_F     == (outside) " << normx_outside
              << " == (inside) " << normx_inside << " == (diff) "
              << abs(normx_outside - normx_inside) << std::endl;

    ///////////////////////////////////////////////////////////////////////////////

    lambda = 1e-2;

    lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), x);

    for (idx_t j = 0; j < n; ++j)
        d[j] = real(A(j, j));
    for (idx_t j = 0; j < n - 1; ++j)
        e[j] = real(A(j, j + 1));

    elden_elim(lambda, d, e, work, x);

    normx_inside = lange(FROB_NORM, x);

    normr_inside1 = lange(FROB_NORM, work);

    normr_inside =
        sqrt(normr_inside0 * normr_inside0 + normr_inside1 * normr_inside1 -
             lambda * lambda * normx_inside * normx_inside);

    unmlq(LEFT_SIDE, CONJ_TRANS, slice(A, range{0, n - 1}, range{1, n}),
          slice(tauw, range{0, n - 1}), x2);

    tik_check(A_copy, bcopy, lambda, x, &normr_outside, &check_tikhonov);

    normx_outside = lange(FROB_NORM, x);

    std::cout << std::endl;

    std::cout << "lambda = " << lambda << std::endl;

    std::cout << "check_tikhonov = " << check_tikhonov << std::endl;

    std::cout << "normr = ||b-Ax||_F  == (outside) " << normr_outside
              << " == (inside) " << normr_inside << " == (diff) "
              << abs(normr_outside - normr_inside) << std::endl;

    std::cout << "normx = ||x||_F     == (outside) " << normx_outside
              << " == (inside) " << normx_inside << " == (diff) "
              << abs(normx_outside - normx_inside) << std::endl;
}
//------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n, k;

    // Default arguments
    m = 5;
    n = 3;
    k = 2;

    // Init random seed
    srand(3);

    // Set output format
    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    // Execute run for different variable types
    // printf("----------------------------------------------------------\n");
    // printf("run< float  >( %d, %d, %d )", m, n, k);
    // run<float>(m, n, k);
    // printf("----------------------------------------------------------\n");

    // printf("run< double >( %d, %d, %d )", m, n, k);
    // run<double>(m, n, k);
    // printf("----------------------------------------------------------\n");

    // printf("run< long double >( %d, %d, %d )", m, n, k);
    // run<long double>(m, n, k);
    // printf("----------------------------------------------------------\n");

    // printf("run< complex<float> >( %d, %d, %d )", m, n, k);
    // run<std::complex<float>>(m, n, k);
    // printf("----------------------------------------------------------\n");

    printf("\nrun< complex<double> >( %d, %d, %d )", m, n, k);
    run<std::complex<double>>(m, n, k);
    printf("----------------------------------------------------------\n");
    return 0;
}
