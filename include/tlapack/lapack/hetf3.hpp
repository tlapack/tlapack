/// @file hetf3.hpp Computes a partial factorization of a symmetric or Hermitian
/// matrix A using the Bunch-Kaufman diagonal pivoting method with level 3 BLAS
/// operations.
/// @author Hugh M Kadhem, University of California Berkeley, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HETF3_HH
#define TLAPACK_HETF3_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/copy.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/her2k.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/syr2k.hpp"
#include "tlapack/lapack/hetrf_blocked.hpp"
#include "tlapack/lapack/rscl.hpp"

namespace tlapack {
/// @brief Options struct for hetrf_blocked()
struct BlockedLDLOpts : public EcOpts {
    constexpr BlockedLDLOpts(const EcOpts& opts = {}) : EcOpts(opts){};

    size_t nb = 32;  ///< Block size
    Op invariant = Op::Trans;
};

/** Computes the partial factorization of a symmetric or Hermitian matrix A
 * using the Bunch-Kaufman diagonal pivoting method with level 3 BLAS
 * operations.
 *
 *      This algorithm writes a consecutive leading/trailing block of $P_i$,
 *      $L_i$ and $D$ factors of the Bunch-Kaufman factorization.
 *
 * @copybrief hetrf_work()
 *
 *      - A is expected to be a leading/trailing view into the whole matrix
 *      being factorized, depending on uplo = Upper/Lower.
 *
 *      - The size of the block factored is either nb or nb-1.
 *      The latter case is indicated by setting the leading element of ipiv to
 *      $n+s$, where $s$ is the index of the column where pivoting stopped.
 *
 * @copydetails hetrf_work()
 *
 * @ingroup computational
 */
template <TLAPACK_UPLO uplo_t,
          TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR ipiv_t,
          TLAPACK_WORKSPACE work_t>
int hetf3(uplo_t uplo,
          matrix_t& A,
          ipiv_t& ipiv,
          work_t& work,
          const BlockedLDLOpts& opts)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Constants
    const idx_t n = nrows(A);
    const idx_t nb = opts.nb;
    const bool hermitian = Op::ConjTrans == opts.invariant;
    // Initialize ALPHA for use in choosing pivot block size.
    const real_t alpha = (real_t(1) + sqrt(real_t(17))) / real_t(8);

    // check arguments
    tlapack_check(uplo == Uplo::Lower || uplo == Uplo::Upper);
    tlapack_check(nrows(A) == ncols(A));
    tlapack_check(nrows(A) == size(ipiv));
    tlapack_check(opts.invariant == Op::Trans ||
                  opts.invariant == Op::ConjTrans);

    // Quick return
    if (n <= 0) return 0;

    // These are QoL wrappers for passing non-const r-value slice references to
    // some functions, to avoid temporary variable declaration clutter.
    // TODO: Add overloaded definitions to the original functions and remove
    // these workarounds.
    constexpr auto copy_ = [](auto&& x, auto&& y) {
        return tlapack::copy(x, y);
    };
    constexpr auto gemv_ = [](auto&& trans, auto&& alpha, auto&& A, auto&& x,
                              auto&& beta, auto&& y) {
        return tlapack::gemv(trans, alpha, A, x, beta, y);
    };
    constexpr auto swap_ = [](auto&& x, auto&& y) {
        return tlapack::swap(x, y);
    };
    constexpr auto rscl_ = [](auto&& alpha, auto&& x) {
        return tlapack::rscl(alpha, x);
    };

    // This is a helper for conjugating a vector as LACGV is currently
    // unimplemented.
    auto conjv = [](auto&& x) {
        const idx_t n = size(x);
        for (int i = 0; i < n; ++i)
            x[i] = conj(x[i]);
    };

    int info = 0;
    if (uplo == Uplo::Upper) {
        // Factorize the trailing nb columns of A using its upper triangle and
        // working backwards, while computing the matrix W = U12*D. which will
        // be used to update the upper-left block A11 of A.
        auto [W, work2] = reshape(work, n, nb);
        int jn = min((int)nb, (int)n);
        // s is the source column of the swap P_i
        // We will use it as the main induction variable and it will decrement
        // by 1 or 2 depending on the rank of each pivot.
        int s;
        for (s = n - 1; s > (int)(n)-jn; --s) {
            // piv will be the target column of the swap P_i
            int piv;
            // j is the trailing column of the pivot
            int j = s;
            // jw is the column of W corresponding to the column j of A.
            int jW = (int)(nb) + j - (int)(n);
            auto Aj0 = slice(A, range{0, j + 1}, range{0, n});
            auto Wj0 = slice(W, range{0, j + 1}, range{0, nb});
            // Copy column w of A to column jW of W and update it.
            copy_(col(Aj0, j), col(Wj0, jW));
            if (hermitian) W(j, jW) = real(W(j, jW));
            if (j + 1 < n) {  // update
                if (hermitian) {
                    copy_(slice(W, j, range{jW + 1, nb}),
                          slice(W, range{j + 1, n}, jW));
                    conjv(slice(W, range{j + 1, n}, jW));
                    gemv_(NO_TRANS, T(-1), cols(Aj0, range{j + 1, n}),
                          slice(W, range{j + 1, n}, jW), T(1), col(Wj0, jW));
                    W(j, jW) = real(W(j, jW));
                }
                else {
                    gemv_(NO_TRANS, T(-1), cols(Aj0, range{j + 1, n}),
                          slice(W, j, range{jW + 1, nb}), T(1), col(Wj0, jW));
                }
            }
            // Determine the rank of the pivot and which columns and rows to
            // swap.
            auto abs_Ajj = abs1(W(j, jW));
            // i_colmax is the index of the largest off-diagonal entry of the
            // updated column j.
            auto i_colmax = tlapack::iamax(slice(W, range{0, j}, jW));
            auto colmax = abs1(W(i_colmax, jW));
            if (max(colmax, abs_Ajj) == 0) {
                piv = j;
                info = (info == 0) ? j + 1 : info;
                if (hermitian) A(j, j) = real(A(j, j));
            }
            else {
                if (abs_Ajj >= alpha * colmax) {
                    piv = j;
                }
                else {
                    // Copy and update column i_colmax into column jW-1 of W.
                    copy_(slice(A, range{0, i_colmax + 1}, i_colmax),
                          slice(W, range{0, i_colmax + 1}, jW - 1));
                    copy_(slice(A, i_colmax, range{i_colmax + 1, j + 1}),
                          slice(W, range{i_colmax + 1, j + 1}, jW - 1));
                    if (hermitian) {
                        W(i_colmax, jW - 1) = real(W(i_colmax, jW - 1));
                        conjv(slice(W, range{i_colmax + 1, j + 1}, jW - 1));
                    }
                    if (j + 1 < n) {
                        if (hermitian) {
                            copy_(slice(W, i_colmax, range{jW + 1, nb}),
                                  slice(W, range{j + 1, n}, jW - 1));
                            conjv(slice(W, range{j + 1, n}, jW - 1));
                            gemv_(NO_TRANS, T(-1), cols(Aj0, range{j + 1, n}),
                                  slice(W, range{j + 1, n}, jW - 1), T(1),
                                  col(Wj0, jW - 1));
                            W(i_colmax, jW - 1) = real(W(i_colmax, jW - 1));
                        }
                        else {
                            gemv_(NO_TRANS, T(-1), cols(Aj0, range{j + 1, n}),
                                  slice(W, i_colmax, range{jW + 1, nb}), T(1),
                                  col(Wj0, jW - 1));
                        }
                    }
                    // i_rowmax is the index of the largest off-diagonal entry
                    // of the updated row i_colmax.
                    auto i_rowmax = i_colmax + 1 +
                                    tlapack::iamax(slice(
                                        W, range{i_colmax + 1, j + 1}, jW - 1));
                    auto rowmax = abs1(W(i_rowmax, jW - 1));
                    if (i_colmax > 0) {
                        i_rowmax = tlapack::iamax(
                            slice(W, range{0, i_colmax}, jW - 1));
                        rowmax = max(rowmax, abs1(W(i_rowmax, jW - 1)));
                    }
                    if (abs_Ajj >= alpha * colmax * colmax / rowmax) {
                        piv = j;
                    }
                    else if (abs1(W(i_colmax, jW - 1)) >= alpha * rowmax) {
                        // We will use updated column i_colmax as a rank 1
                        // pivot. Copy it over column jW of W.
                        piv = i_colmax;
                        copy_(col(Wj0, jW - 1), col(Wj0, jW));
                    }
                    else {
                        // We will use updated column i_colmax as a rank 2
                        // pivot. Decrement s to index the new leading column of
                        // the pivot.
                        piv = i_colmax;
                        --s;
                    }
                }
                if (piv != s) {
                    // Swap rows and columns s and piv.
                    // Their symmetric storage intersects so care must be taken
                    // not to overwrite elements out of order. First copy
                    // non-updated column s of A to column piv of S.
                    A(piv, piv) = hermitian ? real(A(s, s)) : A(s, s);
                    copy_(slice(A, range{piv + 1, s}, s),
                          slice(A, piv, range{piv + 1, s}));
                    if (hermitian) conjv(slice(A, piv, range{piv + 1, s}));
                    if (piv > 0)
                        copy_(slice(A, range{0, piv}, s),
                              slice(A, range{0, piv}, piv));
                    // Swap the non-updated rows,
                    // except the block diagonal which will be later
                    // overwritten.
                    if (j + 1 < n)
                        swap_(slice(A, piv, range{j + 1, n}),
                              slice(A, s, range{j + 1, n}));
                    // Swap the updated rows in W.
                    int sW = (int)(nb) + s - (int)(n);
                    swap_(slice(W, piv, range{sW, nb}),
                          slice(W, s, range{sW, nb}));
                }
                if (j == s) {
                    // Rank 1 pivot: column jW of W
                    // now holds the factor U_jD_j.
                    // copy the column from W to the column j of A,
                    // and rescale the diagonal by D_j.
                    copy_(col(Wj0, jW), col(Aj0, j));
                    if (j > 0)
                        rscl_(hermitian ? real(A(j, j)) : A(j, j),
                              slice(A, range{0, j}, j));
                }
                else {
                    // Rank 2 pivot: columns jW-1:jW of W
                    // now hold the factor W_j = U_jD_j,
                    // where D_j is a symmetric/Hermitian 2-by-2 block.
                    // Write W_jD_j^{-1} onto the columns j-1:j of A,
                    // and copy the upper triangle of D_j onto the diagonal of
                    // A. We use an optimized 2-by-2 inversion algorithm that
                    // minimizes the number of operations by pulling out factors
                    // to set the antidiagonals to -1.
                    T D21 = W(j - 1, jW);
                    T D11 = W(j, jW) / (hermitian ? conj(D21) : D21);
                    T D22 = W(j - 1, jW - 1) / D21;
                    if (hermitian) {
                        real_t d = real_t(1) / (real(D11 * D22) - real_t(1));
                        D21 = d / D21;
                    }
                    else {
                        T d = T(1) / (D11 * D22 - T(1));
                        D21 = d / D21;
                    }
                    for (int k = 0; k < s; ++k) {
                        A(k, j - 1) = D21 * (D11 * W(k, jW - 1) - W(k, jW));
                        A(k, j) = (hermitian ? conj(D21) : D21) *
                                  (D22 * W(k, jW) - W(k, jW - 1));
                    }
                    A(j, j) = W(j, jW);
                    A(j - 1, j) = W(j - 1, jW);
                    A(j - 1, j - 1) = W(j - 1, jW - 1);
                }
            }
            // Update ipiv record
            if (j == s) {
                // rank 1 pivot swapping j with piv
                ipiv[j] = piv;
            }
            else {
                // Rank 2 pivot swapping s with piv and fixing j.
                // Offset by -1 to avoid overlapping piv == 0 == -0.
                ipiv[j] = ipiv[s] = (-piv) - 1;
            }
        }
        if (s >= (int)n - jn) {
            // The last column to be pivoted was not the last column of the
            // block. Indicate its index in the leading block position of ipiv.
            ipiv[n - jn] = n + s;
        }
        if (s >= 0) {
            // Update A11 with the Schur complement of D:
            //  $A11 = A11 - A12 D^{-1} A12^{op}$
            //  $= A11 - W U12^{op}$.
            auto sW = (int)(nb) + s - (int)(n);
            const auto& U12 = slice(A, range{0, s + 1}, range{s + 1, n});
            const auto& W12 = slice(W, range{0, s + 1}, range{sW + 1, nb});
            auto A11 = slice(A, range{0, s + 1}, range{0, s + 1});
            if (hermitian)
                tlapack::her2k(uplo, NO_TRANS, real_t(-0.5), U12, W12,
                               real_t(1), A11);
            else
                tlapack::syr2k(uplo, NO_TRANS, real_t(-0.5), U12, W12,
                               real_t(1), A11);
        }
        // Put U12 in standard form by partially undoing the swaps done to its
        // rows, in the trailing columns of A, by looping back through them in
        // reverse order.
        for (int j = s + 1; j < n; ++j) {
            int s = j;
            int piv = ipiv[j];
            if (piv < 0) {
                piv = -piv - 1;
                ++j;
            }
            // Swap the trailing columns of rows s and piv of A,
            // excluding the block-diagonal pivot block.
            if ((piv != s) & (j + 1 < n)) {
                swap_(slice(A, piv, range{j + 1, n}),
                      slice(A, s, range{j + 1, n}));
            }
        }
    }
    else {
        // Factorize the leading nb columns of A using its lower triangle and
        // working forwards, while computing the matrix W = L21*D. which will be
        // used to update the lower-right block A22 of A. We proceed in exactly
        // the same way as the Upper case, if A is considered as reflected
        // accross both diagonals. There is no jW index because columns of A and
        // W now count from the same base.
        auto [W, work2] = reshape(work, n, nb);
        int jn = min(nb, n);
        int s;
        for (s = 0; s < jn - 1; ++s) {
            int piv;
            int j = s;
            auto Aj0 = slice(A, range{j, n}, range{0, n});
            auto Wj0 = slice(W, range{j, n}, range{0, nb});
            copy_(col(Aj0, j), col(Wj0, j));
            if (hermitian) W(j, j) = real(W(j, j));
            if (j > 0) {
                if (hermitian) {
                    copy_(slice(W, j, range{0, j}), slice(W, range{0, j}, j));
                    conjv(slice(W, range{0, j}, j));
                    gemv_(NO_TRANS, T(-1), cols(Aj0, range{0, j}),
                          slice(W, range{0, j}, j), T(1), col(Wj0, j));
                    W(j, j) = real(W(j, j));
                }
                else {
                    gemv_(NO_TRANS, T(-1), cols(Aj0, range{0, j}),
                          slice(W, j, range{0, j}), T(1), col(Wj0, j));
                }
            }
            auto abs_Ajj = abs1(W(j, j));
            auto i_colmax =
                j + 1 + tlapack::iamax(slice(W, range{j + 1, n}, j));
            auto colmax = abs1(W(i_colmax, j));
            if (max(colmax, abs_Ajj) == 0) {
                piv = j;
                info = (info == 0) ? j + 1 : info;
                A(j, j) = real(A(j, j));
            }
            else {
                if (abs_Ajj >= alpha * colmax) {
                    piv = j;
                }
                else {
                    copy_(slice(A, i_colmax, range{j, i_colmax}),
                          slice(W, range{j, i_colmax}, j + 1));
                    copy_(slice(A, range{i_colmax, n}, i_colmax),
                          slice(W, range{i_colmax, n}, j + 1));
                    if (hermitian) {
                        W(i_colmax, j + 1) = real(W(i_colmax, j + 1));
                        conjv(slice(W, range{j, i_colmax}, j + 1));
                    }
                    if (j > 0) {
                        if (hermitian) {
                            copy_(slice(W, i_colmax, range{0, j}),
                                  slice(W, range{0, j}, j + 1));
                            conjv(slice(W, range{0, j}, j + 1));
                            gemv_(NO_TRANS, T(-1), cols(Aj0, range{0, j}),
                                  slice(W, range{0, j}, j + 1), T(1),
                                  col(Wj0, j + 1));
                            W(i_colmax, j + 1) = real(W(i_colmax, j + 1));
                        }
                        else {
                            gemv_(NO_TRANS, T(-1), cols(Aj0, range{0, j}),
                                  slice(W, i_colmax, range{0, j}), T(1),
                                  col(Wj0, j + 1));
                        }
                    }
                    auto i_rowmax =
                        j + tlapack::iamax(slice(W, range{j, i_colmax}, j + 1));
                    auto rowmax = abs1(W(i_rowmax, j + 1));
                    if (i_colmax + 1 < n) {
                        i_rowmax = i_colmax + 1 +
                                   tlapack::iamax(
                                       slice(W, range{i_colmax + 1, n}, j + 1));
                        rowmax = max(rowmax, abs1(W(i_rowmax, j + 1)));
                    }
                    if (abs_Ajj >= alpha * colmax * colmax / rowmax) {
                        piv = j;
                    }
                    else if (abs1(W(i_colmax, j + 1)) >= alpha * rowmax) {
                        piv = i_colmax;
                        copy_(col(Wj0, j + 1), col(Wj0, j));
                    }
                    else {
                        ++s;
                        piv = i_colmax;
                    }
                }
                if (piv != s) {
                    A(piv, piv) = A(s, s);
                    copy_(slice(A, range{s + 1, piv}, s),
                          slice(A, piv, range{s + 1, piv}));
                    if (hermitian) conjv(slice(A, piv, range{s + 1, piv}));
                    if (piv + 1 < n)
                        copy_(slice(A, range{piv + 1, n}, s),
                              slice(A, range{piv + 1, n}, piv));
                    if (j > 0)
                        swap_(slice(A, piv, range{0, j}),
                              slice(A, s, range{0, j}));
                    swap_(slice(W, piv, range{0, s + 1}),
                          slice(W, s, range{0, s + 1}));
                }
                if (j == s) {
                    copy_(col(Wj0, j), col(Aj0, j));
                    if (j + 1 < n)
                        rscl_(hermitian ? real(W(j, j)) : W(j, j),
                              slice(A, range{j + 1, n}, j));
                }
                else {
                    T D21 = W(j + 1, j);
                    T D22 = W(j, j) / (hermitian ? conj(D21) : D21);
                    T D11 = W(j + 1, j + 1) / D21;
                    if (hermitian) {
                        real_t d = real_t(1) / (real(D11 * D22) - real_t(1));
                        D21 = d / D21;
                    }
                    else {
                        T d = T(1) / (D11 * D22 - T(1));
                        D21 = d / D21;
                    }
                    for (int k = j + 2; k < n; ++k) {
                        A(k, j) = (hermitian ? conj(D21) : D21) *
                                  (D11 * W(k, j) - W(k, j + 1));
                        A(k, j + 1) = D21 * (D22 * W(k, j + 1) - W(k, j));
                    }
                    A(j, j) = W(j, j);
                    A(j + 1, j) = W(j + 1, j);
                    A(j + 1, j + 1) = W(j + 1, j + 1);
                }
            }
            if (j == s) {
                ipiv[j] = piv;
            }
            else {
                ipiv[j] = ipiv[j + 1] = (-piv) - 1;
            }
        }
        if (s < jn) {
            ipiv[jn - 1] = n + s;
        }
        if (s < n) {
            const auto& L21 = slice(A, range{s, n}, range{0, s});
            const auto& W21 = slice(W, range{s, n}, range{0, s});
            auto A22 = slice(A, range{s, n}, range{s, n});
            if (hermitian)
                tlapack::her2k(uplo, NO_TRANS, real_t(-0.5), L21, W21,
                               real_t(1), A22);
            else
                tlapack::syr2k(uplo, NO_TRANS, real_t(-0.5), L21, W21,
                               real_t(1), A22);
        }
        for (int j = s - 1; j > 0; --j) {
            int s = j;
            int piv = ipiv[j];
            if (piv < 0) {
                piv = (-piv) - 1;
                --j;
            }
            if ((piv != s) && (j > 0)) {
                swap_(slice(A, piv, range{0, j}), slice(A, s, range{0, j}));
            }
        }
    }
    return info;
}

}  // namespace tlapack

#endif  // TLAPACK_HETF3_HH
