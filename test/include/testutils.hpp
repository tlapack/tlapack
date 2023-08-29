/// @file testutils.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
///
/// @brief Utility functions for the unit tests
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TESTUTILS_HH
#define TLAPACK_TESTUTILS_HH

// Definitions
#include "testdefinitions.hpp"

// Matrix market
#include "MatrixMarket.hpp"

// Plugin for debug
#include <tlapack/plugins/debugutils.hpp>

// <T>LAPACK
#include <tlapack/base/utils.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/herk.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/laset.hpp>

#ifndef TLAPACK_BUILD_STANDALONE_TESTS
    #include <catch2/catch_template_test_macros.hpp>
    #include <catch2/generators/catch_generators.hpp>

    /// Skip the current test
    #define SKIP_TEST return
#else
    #include <iostream>
    #include <tuple>

    /// Skip the current test
    #define SKIP_TEST return 0

    // Get first argument of a variadic macro
    #define GET_FIRST_ARG(arg1, ...) arg1

    // Below, it is a solution found in
    // https://stackoverflow.com/a/62984543/5253097
    #define DEPAREN(X) ESC(ISH X)
    #define ISH(...) ISH __VA_ARGS__
    #define ESC(...) ESC_(__VA_ARGS__)
    #define ESC_(...) VAN##__VA_ARGS__
    #define VANISH

namespace tlapack {
namespace catch2 {

    std::string return_scanf(const char*);
    std::string return_scanf(std::string);

    template <class T>
    T return_scanf(T)
    {
        if constexpr (std::is_integral<T>::value) {
            T arg;
            std::cin >> arg;
            return arg;
        }
        else if constexpr (std::is_enum<T>::value) {
            char c;
            std::cin >> c;
            return T(c);
        }

        std::abort();
        std::cout << "Include more cases here!\n";
        return {};
    }
    template <class T, class U>
    pair<T, U> return_scanf(pair<T, U>)
    {
        const T x = return_scanf(T());
        const U y = return_scanf(U());
        return pair<T, U>(x, y);
    }
    template <class... Ts>
    std::tuple<Ts...> return_scanf(std::tuple<Ts...>)
    {
        std::tuple<Ts...> t;
        constexpr size_t N = std::tuple_size<std::tuple<Ts...>>::value;
        if constexpr (N > 0)
            std::get<0>(t) = return_scanf(std::get<0>(std::tuple<Ts...>()));
        if constexpr (N > 1)
            std::get<1>(t) = return_scanf(std::get<1>(std::tuple<Ts...>()));
        if constexpr (N > 2)
            std::get<2>(t) = return_scanf(std::get<2>(std::tuple<Ts...>()));
        if constexpr (N > 3)
            std::get<3>(t) = return_scanf(std::get<3>(std::tuple<Ts...>()));
        if constexpr (N > 4)
            std::get<4>(t) = return_scanf(std::get<4>(std::tuple<Ts...>()));
        if constexpr (N > 5)
            std::get<5>(t) = return_scanf(std::get<5>(std::tuple<Ts...>()));
        if constexpr (N > 6) {
            std::abort();
            std::cout << "Include more cases here!\n";
        }
        return t;
    }
}  // namespace catch2
}  // namespace tlapack

    #define TEMPLATE_TEST_CASE(TITLE, TAGS, ...)              \
        using TestType = DEPAREN(GET_FIRST_ARG(__VA_ARGS__)); \
        int main(const int argc, const char* argv[])

    #define GENERATE(...) \
        tlapack::catch2::return_scanf(GET_FIRST_ARG(__VA_ARGS__))

    #define DYNAMIC_SECTION(...) std::cout << __VA_ARGS__ << std::endl;

    #define REQUIRE(cond)          \
        std::cout << #cond << ": " \
                  << (static_cast<bool>(cond) ? "true" : "false") << std::endl

    #define CHECK(cond)            \
        std::cout << #cond << ": " \
                  << (static_cast<bool>(cond) ? "true" : "false") << std::endl

    #define INFO(...) std::cout << __VA_ARGS__ << std::endl;
    #define UNSCOPED_INFO(...) std::cout << __VA_ARGS__ << std::endl;
#endif

namespace tlapack {

/** Calculates res = Q'*Q - I if m <= n or res = Q*Q' otherwise
 *  Also computes the frobenius norm of res.
 *
 * @return frobenius norm of res
 *
 * @param[in] Q m by n (almost) orthogonal matrix
 * @param[out] res n by n matrix as defined above
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrix_t>
real_type<type_t<matrix_t>> check_orthogonality(matrix_t& Q, matrix_t& res)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    const idx_t m = nrows(Q);
    const idx_t n = ncols(Q);

    tlapack_check(nrows(res) == ncols(res));
    tlapack_check(nrows(res) == min(m, n));

    // res = I
    laset(UPPER_TRIANGLE, (T)0.0, (T)1.0, res);
    if (n <= m) {
        // res = Q'Q - I
        herk(UPPER_TRIANGLE, CONJ_TRANS, (real_t)1.0, Q, (real_t)-1.0, res);
    }
    else {
        // res = QQ' - I
        herk(UPPER_TRIANGLE, NO_TRANS, (real_t)1.0, Q, (real_t)-1.0, res);
    }

    // Compute ||res||_F
    return lanhe(FROB_NORM, UPPER_TRIANGLE, res);
}

/** Calculates ||Q'*Q - I||_F if m <= n or ||Q*Q' - I||_F otherwise
 *
 * @return frobenius norm of error
 *
 * @param[in] Q m by n (almost) orthogonal matrix
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrix_t>
real_type<type_t<matrix_t>> check_orthogonality(matrix_t& Q)
{
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    const idx_t m = min(nrows(Q), ncols(Q));

    std::vector<T> res_;
    auto res = new_matrix(res_, m, m);
    return check_orthogonality(Q, res);
}

/** Calculates res = Q'*A*Q - B and the frobenius norm of res
 *
 * @return frobenius norm of res
 *
 * @param[in] A n by n matrix
 * @param[in] Q n by n unitary matrix
 * @param[in] B n by n matrix
 * @param[out] res n by n matrix as defined above
 * @param[out] work n by n workspace matrix
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrix_t>
real_type<type_t<matrix_t>> check_similarity_transform(
    matrix_t& A, matrix_t& Q, matrix_t& B, matrix_t& res, matrix_t& work)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    tlapack_check(nrows(A) == ncols(A));
    tlapack_check(nrows(Q) == ncols(Q));
    tlapack_check(nrows(B) == ncols(B));
    tlapack_check(nrows(res) == ncols(res));
    tlapack_check(nrows(work) == ncols(work));
    tlapack_check(nrows(A) == nrows(Q));
    tlapack_check(nrows(A) == nrows(B));
    tlapack_check(nrows(A) == nrows(res));
    tlapack_check(nrows(A) == nrows(work));

    // res = Q'*A*Q - B
    lacpy(GENERAL, B, res);
    gemm(CONJ_TRANS, NO_TRANS, (real_t)1.0, Q, A, work);
    gemm(NO_TRANS, NO_TRANS, (real_t)1.0, work, Q, (real_t)-1.0, res);

    // Compute ||res||_F
    return lange(FROB_NORM, res);
}

/** Calculates ||Q'*A*Q - B||
 *
 * @return frobenius norm of res
 *
 * @param[in] A n by n matrix
 * @param[in] Q n by n unitary matrix
 * @param[in] B n by n matrix
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrix_t>
real_type<type_t<matrix_t>> check_similarity_transform(matrix_t& A,
                                                       matrix_t& Q,
                                                       matrix_t& B)
{
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    const idx_t n = ncols(A);

    std::vector<T> res_;
    auto res = new_matrix(res_, n, n);
    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);

    return check_similarity_transform(A, Q, B, res, work);
}

//
// GDB doesn't handle templates well, so we explicitly define some versions of
// the functions for common template arguments
//
void print_matrix_r(const LegacyMatrix<float, size_t, Layout::ColMajor>& A);
void print_matrix_d(const LegacyMatrix<double, size_t, Layout::ColMajor>& A);
void print_matrix_c(
    const LegacyMatrix<std::complex<float>, size_t, Layout::ColMajor>& A);
void print_matrix_z(
    const LegacyMatrix<std::complex<double>, size_t, Layout::ColMajor>& A);
void print_rowmajormatrix_r(
    const LegacyMatrix<float, size_t, Layout::RowMajor>& A);
void print_rowmajormatrix_d(
    const LegacyMatrix<double, size_t, Layout::RowMajor>& A);
void print_rowmajormatrix_c(
    const LegacyMatrix<std::complex<float>, size_t, Layout::RowMajor>& A);
void print_rowmajormatrix_z(
    const LegacyMatrix<std::complex<double>, size_t, Layout::RowMajor>& A);

//
// GDB doesn't handle templates well, so we explicitly define some versions of
// the functions for common template arguments
//
std::string visualize_matrix_r(
    const LegacyMatrix<float, size_t, Layout::ColMajor>& A);
std::string visualize_matrix_d(
    const LegacyMatrix<double, size_t, Layout::ColMajor>& A);
std::string visualize_matrix_c(
    const LegacyMatrix<std::complex<float>, size_t, Layout::ColMajor>& A);
std::string visualize_matrix_z(
    const LegacyMatrix<std::complex<double>, size_t, Layout::ColMajor>& A);
std::string visualize_rowmajormatrix_r(
    const LegacyMatrix<float, size_t, Layout::RowMajor>& A);
std::string visualize_rowmajormatrix_d(
    const LegacyMatrix<double, size_t, Layout::RowMajor>& A);
std::string visualize_rowmajormatrix_c(
    const LegacyMatrix<std::complex<float>, size_t, Layout::RowMajor>& A);
std::string visualize_rowmajormatrix_z(
    const LegacyMatrix<std::complex<double>, size_t, Layout::RowMajor>& A);

}  // namespace tlapack

#endif  // TLAPACK_TESTUTILS_HH
