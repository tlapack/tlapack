#include "tik_bidiag_elden.hpp"
#include "tik_qr.hpp"
#include "tik_svd.hpp"
#include "tlapack/base/utils.hpp"

using namespace tlapack;

/// @brief Variants of the algorithm to compute the Cholesky factorization.
enum class TikVariant : char { QR = 'Q', Elden = 'E', SVD = 'S' };

struct TikOpts {
    TikVariant variant = TikVariant::QR;

    constexpr TikOpts(TikVariant v = TikVariant::QR) : variant(v) {}
};

/// Solves tikhonov regularized least squares using QR factorization
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t>
void tkhnv(matrixA_t& A, matrixb_t& b, real_t lambda, const TikOpts& opts = {})
{
    tlapack_check(nrows(A) >= ncols(A));

    if (opts.variant == TikVariant::QR)
        tik_qr(A, b, lambda);
    else if (opts.variant == TikVariant::Elden)
        tik_bidiag_elden(A, b, lambda);
    else
        tik_svd(A, b, lambda);
}
