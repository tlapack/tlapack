// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/getri.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Cauchy matrix properties",
                   "[getri][cauchy]", 
                   TLAPACK_TYPES_TO_TEST) 
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;  // equivalent to using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

// Function to check if a matrix is involutory
template <typename T>
bool isInvolutory(const std::vector<std::vector<T>>& B) {
    size_t n = B.size();
    T identityScaleFactor = pow(2, n - 1);

    // Calculate B^2
    std::vector<std::vector<T>> result(n, std::vector<T>(n, 0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                result[i][j] += B[i][k] * B[k][j];
            }
            if (i == j) {
                result[i][j] /= identityScaleFactor; // Scale the diagonal
            }
        }
    }

    // Check if B^2 is close to the identity matrix
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if ((i == j && std::abs(result[i][j] - 1.0) > 1e-6) || (i != j && std::abs(result[i][j]) > 1e-6)) {
                return false;
            }
        }
    }

    return true;
}}

int main() {
    size_t n = 4; // Change the value of n as needed

    // Generate binomial matrix
    std::vector<std::vector<double>> A;
    binomialMatrix(A, n);

    // Print the binomial matrix A
    std::cout << "Binomial Matrix A:" << std::endl;
    for (const auto& row : A) {
        for (double entry : row) {
            std::cout << entry << " ";
        }
        std::cout << std::endl;
    }

    // Scale A by 2^((1-n)/2) to get B
    double scaleFactor = pow(2, (1 - n) / 2.0);
    std::vector<std::vector<double>> B = A; // Copy A to B
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            B[i][j] *= scaleFactor;
        }
    }

    // Check if B is involutory
    bool isInvolutoryMatrix = isInvolutory(B);

    // Print the result
    if (isInvolutoryMatrix) {
        std::cout << "Matrix B = A * 2^((1-n)/2) is involutory." << std::endl;
    } else {
        std::cout << "Matrix B = A * 2^((1-n)/2) is not involutory." << std::endl;
    }

    return 0;
}