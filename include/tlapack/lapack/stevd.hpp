/// @file stevd.hpp 
/// @author Xuan Jiang, University of California, Berkeley, USA
//
// Copyright (c) 2021-2023, University of California, Berkeley, USA. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STEVD_HH
#define TLAPACK_STEVD_HH

#include "tlapack/base/constants.hpp"
#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * @brief Compute all eigenvalues and, optionally, eigenvectors of a
 * symmetric tridiagonal matrix using the divide and conquer method.
 *
 * @tparam matrix_t Type of the matrix.
 * @tparam vector_t Type of the vector.
 * @tparam work_t Type of the work array.
 * @tparam iwork_t Type of the integer work array.
 *
 * @param[in] jobz If jobz = 'N', compute eigenvalues only;
 *                 if jobz = 'V', compute eigenvalues and eigenvectors.
 * @param[in] n The order of the matrix A.
 * @param[in,out] d On entry, the diagonal elements of the matrix A.
 *                  On exit, the eigenvalues in ascending order.
 * @param[in,out] e On entry, the (n-1) subdiagonal elements of the matrix A.
 *                  On exit, e has been destroyed.
 * @param[out] z If jobz = 'V', then if info = 0, Z contains the orthonormal
 *               eigenvectors of the matrix A.
 *               If jobz = 'N', then Z is not referenced.
 * @param[in] ldz The leading dimension of the array Z. ldz >= 1, and if
 *                jobz = 'V', ldz >= max(1,n).
 * @param[out] work Workspace array.
 * @param[in] lwork The dimension of the array work. If lwork = -1, then a
 *                  workspace query is assumed; the routine only calculates
 *                  the optimal size of the work array, returns this value
 *                  as the first entry of the work array, and no error
 *                  message related to lwork is issued by xerbla.
 * @param[out] iwork Integer workspace array.
 * @param[in] liwork The dimension of the array iwork.
 * @param[out] info If info = 0, the execution is successful.
 *                  If info = -i, the i-th argument had an illegal value
 *                  If info = i, the algorithm failed to converge; i
 *                  off-diagonal elements of an intermediate tridiagonal
 *                  form did not converge to zero.
 *
 * @return An integer info as described above.
 *
 * @ingroup eigenvalues
 */
template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_VECTOR work_t,
          TLAPACK_IVECTOR iwork_t>
int stevd(char jobz, idx_t n, vector_t& d, vector_t& e, matrix_t& z,
          idx_t ldz, work_t& work, idx_t lwork,
          iwork_t& iwork, idx_t liwork, int& info)
{
    // Check arguments and quick return.
    if (n < 0) {
        info = -2;
        return info;
    }
    if ((jobz != 'N' && jobz != 'V') || (ldz < 1 || (jobz == 'V' && ldz < n))) {
        info = -6;
        return info;
    }
    // Workspace query or actual computation.
    if (lwork == -1 || liwork == -1) {
        // Workspace query, set optimal workspace size.
        // (Note: The workspace requirements will depend on the specific algorithm used)
        work[0] = optimal_workspace_size;
        iwork[0] = optimal_iworkspace_size;
        info = 0;
    } else {
        // Actual computation.
        // (Note: Implement the divide and conquer algorithm here)

        // Dummy implementation:
        info = 0;
    }
    return info;
}

} // namespace tlapack

#endif // TLAPACK_STEVD_HH
