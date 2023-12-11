#include <iostream>
#include <vector>
#include <cmath>
#include <testutils.hpp>
using namespace tlapack;

    // using matrix_t = TestType;
    // using T = type_t<matrix_t>;
    // using idx_t = size_type<matrix_t>;

    // Function for calculating the binomial coefficient C(n, k)
    unsigned long long binomialCoeff(int n, int k) {
        if (k == 0 || k == n) {
            return 1;
        } else {
            return binomialCoeff(n - 1, k - 1) + binomialCoeff(n - 1, k);
        }
    }

    // Function for checking whether a matrix is a binomial matrix
    template <typename T>
    bool isBinomialMatrix(const std::vector<std::vector<T>>& A, double tol = 1e-10) {
        int n = A.size();

        // Check whether A is identical to its inverse value, taking the tolerance into account
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                T expected_value = static_cast<T>(binomialCoeff(n - 1, j) * binomialCoeff(n - 1, i - 1));
                if (std::abs(A[i][j] - expected_value) > tol) {
                    return false;
                }
            }
        }
        return true;
    }
TEMPLATE_TEST_CASE("is_eigen_dense, is_eigen_block and is_eigen_map work", "[plugins]", TLAPACK_TYPES_TO_TEST)
{
    int n = GENERATE(3, 5, 7, 9, 11);

    DYNAMIC_SECTION("n = " <<n)
    {
    std::vector<std::vector<int>> binomial_matrix_result;
    // ... Populate binomial_matrix_result with BinomialMatrix function

    // Check whether the matrix created is a binomial matrix
    bool is_binomial = isBinomialMatrix(binomial_matrix_result, 1e-10);

    std::cout << "Dimension n = " << n << std::endl;
    std::cout << "Is Binomialmatrix: " << (is_binomial ? "true" : "false") << std::endl;
    std::cout << "------------------------------" << std::endl;
    CHECK(is_binomial);
    }
}
// int main() {
//     // Test für verschiedene Dimensionen n
//     for (int n = 2; n <= 9; ++n) {
//         std::vector<std::vector<int>> binomial_matrix_result;
//         // ... Populate binomial_matrix_result with BinomialMatrix function

//         // Überprüfen, ob die erstellte Matrix eine Binomialmatrix ist
//         bool is_binomial = isBinomialMatrix(binomial_matrix_result, 1e-10);

//         std::cout << "Dimension n = " << n << std::endl;
//         std::cout << "Ist Binomialmatrix: " << (is_binomial ? "true" : "false") << std::endl;
//         std::cout << "------------------------------" << std::endl;
//         CHECK(is_binomial);
//     }

//     return 0;
// }
