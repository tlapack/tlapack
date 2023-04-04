#include "testutils.hpp"

namespace tlapack {

//
// GDB doesn't handle templates well, so we explicitly define some versions of
// the functions for common template arguments
//
void print_matrix_r(const legacyMatrix<float, size_t, Layout::ColMajor>& A)
{
    print_matrix(A);
}
void print_matrix_d(const legacyMatrix<double, size_t, Layout::ColMajor>& A)
{
    print_matrix(A);
}
void print_matrix_c(
    const legacyMatrix<std::complex<float>, size_t, Layout::ColMajor>& A)
{
    print_matrix(A);
}
void print_matrix_z(
    const legacyMatrix<std::complex<double>, size_t, Layout::ColMajor>& A)
{
    print_matrix(A);
}
void print_rowmajormatrix_r(
    const legacyMatrix<float, size_t, Layout::RowMajor>& A)
{
    print_matrix(A);
}
void print_rowmajormatrix_d(
    const legacyMatrix<double, size_t, Layout::RowMajor>& A)
{
    print_matrix(A);
}
void print_rowmajormatrix_c(
    const legacyMatrix<std::complex<float>, size_t, Layout::RowMajor>& A)
{
    print_matrix(A);
}
void print_rowmajormatrix_z(
    const legacyMatrix<std::complex<double>, size_t, Layout::RowMajor>& A)
{
    print_matrix(A);
}

//
// GDB doesn't handle templates well, so we explicitly define some versions of
// the functions for common template arguments
//
std::string visualize_matrix_r(
    const legacyMatrix<float, size_t, Layout::ColMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_matrix_d(
    const legacyMatrix<double, size_t, Layout::ColMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_matrix_c(
    const legacyMatrix<std::complex<float>, size_t, Layout::ColMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_matrix_z(
    const legacyMatrix<std::complex<double>, size_t, Layout::ColMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_rowmajormatrix_r(
    const legacyMatrix<float, size_t, Layout::RowMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_rowmajormatrix_d(
    const legacyMatrix<double, size_t, Layout::RowMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_rowmajormatrix_c(
    const legacyMatrix<std::complex<float>, size_t, Layout::RowMajor>& A)
{
    return visualize_matrix(A);
}
std::string visualize_rowmajormatrix_z(
    const legacyMatrix<std::complex<double>, size_t, Layout::RowMajor>& A)
{
    return visualize_matrix(A);
}

}  // namespace tlapack