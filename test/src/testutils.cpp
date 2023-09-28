#include "testutils.hpp"

namespace tlapack {

namespace catch2 {
    std::string return_scanf(const char*)
    {
        std::string str;
        std::cin >> str;
        return str;
    }
    std::string return_scanf(std::string) { return return_scanf(""); }
}  // namespace catch2

//
// GDB doesn't handle templates well, so we explicitly define some versions of
// the functions for common template arguments
//
void print_matrix_half(
    const Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic>& A)
{
    print_matrix(A);
}
void print_matrix_r(const LegacyMatrix<float, size_t, Layout::ColMajor>& A)
{
    print_matrix(A);
}
void print_matrix_d(const LegacyMatrix<double, size_t, Layout::ColMajor>& A)
{
    print_matrix(A);
}
void print_matrix_c(
    const LegacyMatrix<std::complex<float>, size_t, Layout::ColMajor>& A)
{
    print_matrix(A);
}
void print_matrix_z(
    const LegacyMatrix<std::complex<double>, size_t, Layout::ColMajor>& A)
{
    print_matrix(A);
}
void print_rowmajormatrix_r(
    const LegacyMatrix<float, size_t, Layout::RowMajor>& A)
{
    print_matrix(A);
}
void print_rowmajormatrix_d(
    const LegacyMatrix<double, size_t, Layout::RowMajor>& A)
{
    print_matrix(A);
}
void print_rowmajormatrix_c(
    const LegacyMatrix<std::complex<float>, size_t, Layout::RowMajor>& A)
{
    print_matrix(A);
}
void print_rowmajormatrix_z(
    const LegacyMatrix<std::complex<double>, size_t, Layout::RowMajor>& A)
{
    print_matrix(A);
}

//
// GDB doesn't handle templates well, so we explicitly define some versions of
// the functions for common template arguments
//
std::string visualize_matrix_r(
    const LegacyMatrix<float, size_t, Layout::ColMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_matrix_d(
    const LegacyMatrix<double, size_t, Layout::ColMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_matrix_c(
    const LegacyMatrix<std::complex<float>, size_t, Layout::ColMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_matrix_z(
    const LegacyMatrix<std::complex<double>, size_t, Layout::ColMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_rowmajormatrix_r(
    const LegacyMatrix<float, size_t, Layout::RowMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_rowmajormatrix_d(
    const LegacyMatrix<double, size_t, Layout::RowMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_rowmajormatrix_c(
    const LegacyMatrix<std::complex<float>, size_t, Layout::RowMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_rowmajormatrix_z(
    const LegacyMatrix<std::complex<double>, size_t, Layout::RowMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_matrix_half(
    const Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic>& A)
{
    return visualize_matrix(A);
}
std::string visualize_matrix_chalf(
    const Eigen::
        Matrix<std::complex<Eigen::half>, Eigen::Dynamic, Eigen::Dynamic>& A)
{
    return visualize_matrix(A);
}

std::string visualize_vector_r(const std::vector<float>& v)
{
    return visualize_vector(v);
}
std::string visualize_vector_d(const std::vector<double>& v)
{
    return visualize_vector(v);
}
std::string visualize_vector_c(const std::vector<std::complex<float>>& v)
{
    return visualize_vector(v);
}
std::string visualize_vector_z(const std::vector<std::complex<double>>& v)
{
    return visualize_vector(v);
}
std::string visualize_vector_half(const std::vector<Eigen::half>& v)
{
    return visualize_vector(v);
}
std::string visualize_vector_chalf(
    const std::vector<std::complex<Eigen::half>>& v)
{
    return visualize_vector(v);
}

}  // namespace tlapack