// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"


#include <tlapack/lapack/gesvd.hpp>

using namespace tlapack;

// Utility function to check if elements of a vector are in increasing order.
template <typename vector_t>
bool isIncreasing(const vector_t& vec) {
    for (std::size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] <= vec[i-1]) {
            return false;
        }
    }
    return true;
}

TEMPLATE_TEST_CASE("Cauchy matrix properties", "[svd][cauchy]", TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    const idx_t n = GENERATE(3, 5, 10, 32);

    // Define the matrices and vectors
    std::vector<T> x(n);
    std::vector<T> y(n);
    std::vector<T> C_;
    auto C = new_matrix(C_, n, n);

    for (idx_t i = 0; i < n; ++i) {
        x[i] = (T)(i+1);
        y[i] = (T)(n + i + 1);
    }

    generateCauchy(C, x, y);

    DYNAMIC_SECTION()
    {
        // What I want to test:
        // include/tlapack/lapack/getri.hpp
        // C^−1 = V Σ^−1 U^T (Can I find the inverse in another way?)
        // Explicit formula for C: 
        // π(x) = product of x from k to n
        // fraq = (x_j + y_i) * π_(k≠j)(x_j - x_k) * π_(k≠i)(y_i - y_k)
        // (C^-1_ij) = π[(x_j + y_k) * (x_k + y_i)] / fraq
        // Do this for all i,j => get C^-1

        // Calculate the difference in norm of the two matrices,
        // which I expect to be small (machine eps???)
    }
}