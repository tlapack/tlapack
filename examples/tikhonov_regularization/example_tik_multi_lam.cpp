//
// Example on how to solve a Tikhonov regularized least squares problem with
// multiple λ's using Eldén's bidiagonalization algorithm. The problem is
// first "pushed" to bidiagonal form. Then a λ is tried. After this we have
// access to ||x||₂ and ||b-Ax||₂ and can decide on a new λ. And repeat
// until a suitable λ is found. Once a suitable λ is found, a last
// transformation is done on x to "pop" the solution to the original problem
// (the one not in bidiagonal form).
//
// We use Eldén's Tikhonov regularized least squares problem for bidiagonal
// matrix algorithm.
//
// Other methods would be
// (*) Normal Equations (and Cholesky) which is unstable;
// (*) QR factorization which requires many more computations for each λ;
// (*) SVD, which is great once you have it, and enable to quickly go over many
// λ's but is more expensive than bidiagonalization to obtain.

#include <tlapack/plugins/legacyArray.hpp>
//
#include <tlapack/lapack/bidiag.hpp>
#include <tlapack/lapack/elden_elim.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/unmlq.hpp>
#include <tlapack/lapack/unmqr.hpp>

#include "../../test/include/MatrixMarket.hpp"
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
    auto view_x = slice(x, range{1, n}, range{0, k});

    real_t lambda;

    real_t normr_outside;
    real_t check_tikhonov;
    real_t normx_inside;
    real_t normr_inside1;
    real_t normr_inside;
    real_t normx_outside;

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

    // push A and b to bidiagonal world
    bidiag(A, tauv, tauw);

    unmqr(LEFT_SIDE, CONJ_TRANS, A, tauv, b);

    // will be useful to compute ||b-Ax||₂ later on
    real_t normr_inside0 = lange(FROB_NORM, slice(b, range{n, m}, range{0, k}));

    ///////////////////////////////////////////////////////////////////////////////
    // try a first λ

    lambda = 1e-2;

    lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), x);

    for (idx_t j = 0; j < n; ++j)
        d[j] = real(A(j, j));
    for (idx_t j = 0; j < n - 1; ++j)
        e[j] = real(A(j, j + 1));

    elden_elim(lambda, d, e, x, work);

    normx_inside = lange(FROB_NORM, x);

    normr_inside1 = lange(FROB_NORM, work);

    normr_inside =
        sqrt(normr_inside0 * normr_inside0 + normr_inside1 * normr_inside1 -
             lambda * lambda * normx_inside * normx_inside);

    // at this point we have ||b-Ax||₂ and ||x||₂ and can decide for a new
    // λ the following lines are not needed and should not be used in
    // general, these are checks, please resume at "trying a new λ"

    // ↓↓↓ checks -- omit in real-life applications ↓↓↓
    unmlq(LEFT_SIDE, CONJ_TRANS, slice(A, range{0, n - 1}, range{1, n}),
          slice(tauw, range{0, n - 1}), view_x);

    tik_check(A_copy, bcopy, lambda, x, &normr_outside, &check_tikhonov);

    normx_outside = lange(FROB_NORM, x);

    std::cout << std::endl;

    std::cout << "λ = " << lambda << std::endl;

    std::cout << "check_tikhonov = " << check_tikhonov << std::endl;

    std::cout << "normr = ||b-Ax||_F  == (outside) " << normr_outside
              << " == (inside) " << normr_inside << " == (diff) "
              << abs(normr_outside - normr_inside) << std::endl;

    std::cout << "normx = ||x||_F     == (outside) " << normx_outside
              << " == (inside) " << normx_inside << " == (diff) "
              << abs(normx_outside - normx_inside) << std::endl;
    // ↑↑↑ checks -- omit in real-life applications ↑↑↑

    ///////////////////////////////////////////////////////////////////////////////
    // trying a new λ

    lambda = 1e-4;

    lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), x);

    for (idx_t j = 0; j < n; ++j)
        d[j] = real(A(j, j));
    for (idx_t j = 0; j < n - 1; ++j)
        e[j] = real(A(j, j + 1));

    elden_elim(lambda, d, e, x, work);

    normx_inside = lange(FROB_NORM, x);

    normr_inside1 = lange(FROB_NORM, work);

    normr_inside =
        sqrt(normr_inside0 * normr_inside0 + normr_inside1 * normr_inside1 -
             lambda * lambda * normx_inside * normx_inside);

    // at this point we have ||b-Ax||₂ and ||x||₂ and we can decide for a new
    // λ we assume that we are satisfied with our λ and our "x", so
    // the next line is needed to "pop" back x to the non-bidiagonal world.

    unmlq(LEFT_SIDE, CONJ_TRANS, slice(A, range{0, n - 1}, range{1, n}),
          slice(tauw, range{0, n - 1}), view_x);

    // ↓↓↓ checks -- omit in real-life applications ↓↓↓
    tik_check(A_copy, bcopy, lambda, x, &normr_outside, &check_tikhonov);

    normx_outside = lange(FROB_NORM, x);

    std::cout << std::endl;

    std::cout << "λ = " << lambda << std::endl;

    std::cout << "check_tikhonov = " << check_tikhonov << std::endl;

    std::cout << "normr = ||b-Ax||_F  == (outside) " << normr_outside
              << " == (inside) " << normr_inside << " == (diff) "
              << abs(normr_outside - normr_inside) << std::endl;

    std::cout << "normx = ||x||_F     == (outside) " << normx_outside
              << " == (inside) " << normx_inside << " == (diff) "
              << abs(normx_outside - normx_inside) << std::endl;
    // ↑↑↑ checks -- omit in real-life applications ↑↑↑
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
