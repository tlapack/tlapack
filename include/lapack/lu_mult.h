#ifndef LU_MULT
#define LU_MULT

using namespace tlapack;

template <typename idx_t>
struct lu_mult_opts_t
{
    // Optimization parameter. Matrices smaller than nx will not
    // be multiplied using recursion. Must be at least 1.
    idx_t nx = 1;
};

/**
 *
 * @brief in-place multiplication of lower triangular matrix L and upper triangular matrix U.
 *
 * @param[in,out] A n-by-n matrix
 *      On entry, the strictly lower triangular entries of A contain the matrix L.
 *      L is assumed to have unit diagonal.  
 *      The upper triangular entires of A contain the matrix U. 
 *      On exit, A contains the product L*U. 
 * @param[in] (Optional)  struct containing optimization parameters. See lu_mult_ops_t
 * 
 * @ingroup util
 */
template <typename matrix_t>
void lu_mult(matrix_t &A, const lu_mult_opts_t<size_type<matrix_t>> &opts = {})
{
    using idx_t = size_t;
    using T = type_t<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    using real_t = real_type<T>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (n <= opts.nx)
    { // Matrix is small, use for loops instead of recursion
        for (idx_t i2 = n; i2 > 0; --i2)
        {
            idx_t i = i2 - 1;
            for (idx_t j2 = n; j2 > 0; --j2)
            {
                idx_t j = j2 - 1;
                auto sum = make_scalar<T>(0, 0);
                for (idx_t k = 0; k <= min(i, j); ++k)
                {
                    if (i == k)
                        sum += A(k, j);
                    else
                        sum += A(i, k) * A(k, j);
                }
                A(i, j) = sum;
            }
        }
        return;
    }

    idx_t n0 = n / 2;

    auto A00 = tlapack::slice(A, range(0, n0), range(0, n0));
    auto A01 = tlapack::slice(A, range(0, n0), range(n0, n));
    auto A10 = tlapack::slice(A, range(n0, n), range(0, n0));
    auto A11 = tlapack::slice(A, range(n0, n), range(n0, n));

    lu_mult(A11, opts);

    // Step 2
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(1), A10, A01, T(1), A11);

    // Step 3
    tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower, tlapack::Op::NoTrans,
                  tlapack::Diag::Unit, real_t(1), A00, A01);

    // Step 4
    tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper, tlapack::Op::NoTrans,
                  tlapack::Diag::NonUnit, real_t(1), A00, A10);

    // Step 5
    lu_mult(A00, opts);

    return;
}

#endif // LU_MULT