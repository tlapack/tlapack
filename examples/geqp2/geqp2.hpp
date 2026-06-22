
// The Q from Businger and Golub algorithm can be very inaccurate, even for
// small matrices. Th reconstruction error for drmac is wrong

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/copy.hpp>
#include <tlapack/blas/syrk.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/larf.hpp>
#include <tlapack/lapack/larfg.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>  //get rid of?

// C++ headers
#include <algorithm>
#include <chrono>  // for high_resolution_clock
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>


//------------------------------------------------------------------------------
// check function used in Stewarts algorithm
template <typename matrixA_t,
          typename matrixV_t,
          typename vectorT_t,
          typename vectorP_t>
inline void check(matrixA_t& A,
                  matrixV_t& V,
                  const vectorT_t& tau,
                  const vectorP_t& perm,
                  const std::string& matrix_name)

{
    using T = typename tlapack::type_t<matrixA_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrixA_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    // Functors for creating new matrices
    tlapack::Create<matrixA_t> new_matrix;

    bool verbose = false;

    // Permute the columns of A according to the pivot order in perm.
    std::vector<T> A_perm_(m * n);
    auto A_perm = new_matrix(A_perm_, m, n);
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A_perm(i, j) = A(i, perm[j]);
        }
    }

    // Copy the upper-triangular part of V into R.
    // Zero the full matrix first so lower triangle does not contain garbage.
    std::vector<T> R_(n * n);
    auto R = new_matrix(R_, n, n);
    tlapack::laset(tlapack::GENERAL, T(0.0), T(0.0), R);
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, V, R);

    // Form the explicit orthogonal matrix Q from the Householder factors.
    std::vector<T> Q_(m * m);
    auto Q = new_matrix(Q_, m, m);
    tlapack::lacpy(tlapack::GENERAL, V, Q);
    tlapack::ung2r(Q, tau);

    // print the orthogonal Q from qrbg
    if (verbose) {
        std::cout << std::endl << "Q_check =";
        printMatrix(Q);
        std::cout << std::endl;
    }

    // Compute the reconstruction accuracy ||AP - Q*R||.
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(-1.0), Q, R,
                  T(1.0), A_perm);

    real_t recon_error = tlapack::lange(tlapack::FROB_NORM, A_perm);

    // Compute the orthogonality error ||I - Q^H Q||.
    std::vector<T> work_;
    auto work = new_matrix(work_, m, m);
    tlapack::laset(tlapack::GENERAL, T(0.0), T(1.0), work);
    tlapack::gemm(tlapack::Op::Trans, tlapack::Op::NoTrans, T(-1.0), Q, Q,
                  T(1.0), work);
    real_t orthogonality_error = tlapack::lange(tlapack::FROB_NORM, work);

    std::cout << std::endl
              << "Reconstruction error ||AP - Q*R|| for " << matrix_name
              << " = " << recon_error << std::endl;
    std::cout << "Orthogonality error ||I - Q^H Q|| for " << matrix_name
              << " = " << orthogonality_error << std::endl;
}

// Implementation of the laqps
template <typename matrixA_t, typename vectorT_t, typename vectorP_t>
inline void laqps(matrixA_t& A, vectorT_t& tau, vectorP_t& perm)
{
    using T = typename tlapack::type_t<matrixA_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrixA_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    // Machine precision constants
    const real_t eps = std::numeric_limits<real_t>::epsilon();
    const real_t safety_threshold = std::sqrt(eps);  // √e guard from Table 3


    // Tracking norm vectors
    std::vector<real_t> vn1(n);  // Current partial norms (ω)
    std::vector<real_t> vn2(n);  // Original initial column norms (ν)

    // Initialize norms for the active block starting from 'offset'
    for (idx_t j = 0; j < n; ++j) {
        auto A_j = slice(A, range(0, m), j);
        vn1[j] = tlapack::nrm2(A_j);
        vn2[j] = vn1[j];
    }

    // List to accumulate columns experiencing severe cancellations
    std::vector<idx_t> unresolved_columns;

    for (idx_t k = 0; k < std::min(m, n); ++k) {
        // 1. Identify pivot column from non-unresolved active columns
        idx_t pivot = k;
        real_t max_norm = vn1[k];
        for (idx_t j = k + 1; j < n; ++j) {
            if (vn1[j] > max_norm) {
                max_norm = vn1[j];
                pivot = j;
            }
        }

        // 2. Perform swap if necessary
        if (pivot != k) {
            for (idx_t i = 0; i < m; ++i) {
                std::swap(A(i, k), A(i, pivot));
            }
            // std::swap(tau[k], tau[pivot]);
            std::swap(perm[k], perm[pivot]);
            std::swap(vn1[k], vn1[pivot]);
            std::swap(vn2[k], vn2[pivot]);
        }

        // 3. Generate Householder reflector for column k
        auto col_k = slice(A, range(k, m), k);
        tlapack::larfg(tlapack::Direction::Forward, tlapack::StoreV::Columnwise,
                       col_k, tau[k]);

        // 4. Apply reflector matrix to trailing columns
        if (k < n - 1) {
            auto A_trailing = slice(A, range(k, m), range(k + 1, n));
            tlapack::larf(tlapack::Side::Left, tlapack::Direction::Forward,
                          tlapack::StoreV::Columnwise, col_k, tau[k],
                          A_trailing);
        }

        // 5. Update column norms using the Table 3 strategy
        // unresolved_columns.clear();
        for (idx_t j = k + 1; j < n; ++j) {
            if (vn1[j] > static_cast<real_t>(0.0)) {
                // t = |A(k,j)| / vn1(j)
                real_t t = tlapack::abs(A(k, j)) / vn1[j];

                // t = max(0, 1 - t^2)
                t = std::max(static_cast<real_t>(0.0),
                             static_cast<real_t>(1.0) - t * t);

                // t2 = t * (vn1(j) / vn2(j))^2
                real_t norm_ratio = vn1[j] / vn2[j];
                real_t t2 = t * (norm_ratio * norm_ratio);

                // Table 3 Safety Check: if t2 <= √e, push to unresolved list
                if (t2 <= safety_threshold) {
                    auto A_col = slice(A, range(k + 1, m), j);
                    vn1[j] = tlapack::nrm2(A_col);
                    vn2[j] = vn1[j];  // Reset base tracking metric
                    // unresolved_columns.push_back(j);
                }
                else {
                    // Safe to update using the stable scalar formula
                    vn1[j] = vn1[j] * std::sqrt(t);
                }
            }
        }
    }
}

template <typename T>
void run(size_t m, size_t n, size_t r)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Arrays
    n = m;
    r = n;
    std::vector<T> tau(n);
    std::vector<T> tau_drmac(n);
    std::vector<idx_t> perm_laqps(n);
    std::iota(perm_laqps.begin(), perm_laqps.end(), 0);

    std::vector<idx_t> perm_row(m);
    std::iota(perm_row.begin(), perm_row.end(), 0);

    // Matrix
    std::vector<T> A_orig_;
    std::vector<T> A_drmac_;
    auto A_drmac = new_matrix(A_drmac_, m, n);
    auto A_orig = new_matrix(A_orig_, m, n);
    std::vector<T> A_perm_;
    auto A_perm = new_matrix(A_perm_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> C_;

    // Initialize arrays with junk
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A_orig(i, j) = static_cast<T>(0xDEADBEEF);
        }
        tau[j] = static_cast<T>(0xFFBADD11);
        tau_drmac[j] = static_cast<T>(0xFFBADD11);
    }

    tlapack::laset(tlapack::Uplo::General, static_cast<T>(0), static_cast<T>(0),
                   C);
    tlapack::laset(tlapack::Uplo::General, static_cast<T>(0), static_cast<T>(0),
                   D);

    T cosine = static_cast<T>(0.4664999999999993);
    const T eps = std::numeric_limits<T>::epsilon();
    cosine = cosine * (static_cast<T>(1.0) + eps);
    T sine = static_cast<T>(std::sqrt(static_cast<T>(1.0) - cosine * cosine));

    tlapack::laset(tlapack::Uplo::Upper, cosine, static_cast<T>(1.0), D);

    // set the diagonal of C to an increment of (pi/4)^2
    for (size_t i = 0; i < r; ++i)
        C(i, i) = static_cast<T>(pow(sine, i));


    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(1.0), C, D,
                  T(0.0), A_orig);

    std::random_device rd;
    std::mt19937 gen_col(rd());
    std::vector<int> col_permutaton(n);
    std::iota(col_permutaton.begin(), col_permutaton.end(), 0);
    std::shuffle(col_permutaton.begin(), col_permutaton.end(), gen_col);

    std::random_device rand;
    std::mt19937 gen_row(rand());
    std::vector<int> row_permutation(m);
    std::iota(row_permutation.begin(), row_permutation.end(), 0);
    std::shuffle(row_permutation.begin(), row_permutation.end(), gen_row);

    for (idx_t j = 0; j < n; ++j) {
        while (col_permutaton[j] != j) {
            int current_col = col_permutaton[j];
            for (idx_t i = 0; i < m; ++i) {
                std::swap(A_orig(i, j), A_orig(i, current_col));
            }
            std::swap(col_permutaton[j], col_permutaton[current_col]);
        }
    }

    for (idx_t i = 0; i < m; ++i) {
        while (row_permutation[i] != i) {
            int current_row = row_permutation[i];
            for (idx_t j = 0; j < n; ++j) {
                std::swap(A_orig(i, j), A_orig(current_row, j));
            }
            std::swap(row_permutation[i], row_permutation[current_row]);
        }
    }

    // Copy matrix A_orig into separate matrices for each factorization.
    tlapack::lacpy(tlapack::GENERAL, A_orig, A_drmac);

    std::ofstream data_file("qrcp_comps_data.py");
    std::vector<real_t> mu(n);
    std::vector<real_t> r_kk(n);
    auto max_temp = static_cast<real_t>(0.0);

    // Compute the QR factorization of A with column pivoting using Businger and
    // Golub's algorithm.

    laqps(A_drmac, tau_drmac, perm_laqps);
    if (verbose) {
        std::cout << std::endl << "perm_laqps =";
        for (idx_t j = 0; j < n; ++j)
            std::cout << " " << perm_laqps[j];
        std::cout << std::endl;
    }

    for (idx_t i = 0; i < n - 1; ++i)
        mu[i] = static_cast<real_t>(0.0);

    for (idx_t i = 0; i < n - 1; ++i) {
        r_kk[i] = std::abs(A_drmac(i, i));
        for (idx_t j = i + 1; j < n; ++j) {
            max_temp = std::abs(A_drmac(i, j));
            if (max_temp > mu[i]) mu[i] = max_temp;
        }
    }
    mu[n - 1] = std::abs(A_drmac(n - 1, n - 1));

    data_file << "r_kk_drmac = [ ";
    for (idx_t i = 0; i < n; ++i)
        data_file << r_kk[i] << ", ";
    data_file << "]" << std::endl;
    data_file << "mu_drmac = [ ";
    for (idx_t i = 0; i < n; ++i)
        data_file << mu[i] << ", ";
    data_file << "]" << std::endl;

    check(A_orig, A_drmac, tau_drmac, perm_laqps, "Drmac");

    data_file.close();
}

// clean up code, check why reconstruction error is the same, and swap

int main(int argc, char** argv)
{
    int m, n, r;

    // Default arguments
    m = (argc < 2) ? 300 : atoi(argv[1]);
    n = (argc < 3) ? 300 : atoi(argv[2]);
    r = (argc < 4) ? 3 : atoi(argv[3]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< double >( %d, %d, %d )\n", m, n, r);
    run<double>(m, n, r);
    printf("-----------------------\n");

    return 0;
}

//