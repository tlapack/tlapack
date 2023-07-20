/// @file workspace.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_WORKSPACE_HH
#define TLAPACK_WORKSPACE_HH

#include "tlapack/base/exceptionHandling.hpp"
#include "tlapack/base/types.hpp"

namespace tlapack {

/**
 * @brief Workspace
 *
 * The objects of this class always maintain one of the states:
 *      1. `(n > 1) && (m > 0)`.
 *      2. `((n <= 1) || (m == 0)) && (ldim == m)`.
 */
struct Workspace {
    using idx_t = std::size_t;

    // Constructors:

    inline constexpr Workspace(byte* ptr = nullptr, idx_t m = 0, idx_t n = 1)
        : m(m), n(n), ptr(ptr), ldim(m)
    {
        tlapack_check(m >= 0);
        tlapack_check(n >= 0);
    }

    inline constexpr Workspace(byte* ptr, idx_t m, idx_t n, idx_t ldim)
        : m(m), n(n), ptr(ptr), ldim(ldim)
    {
        tlapack_check(m >= 0);
        tlapack_check(n >= 0);

        tlapack_check(ldim >= m);
        if (n <= 1 || m == 0) this->ldim = m;
    }

    template <class T, class idx_t>
    inline constexpr Workspace(const legacy::Matrix<T, idx_t>& A)
        : ptr((byte*)A.ptr), ldim(A.ldim * sizeof(T))
    {
        tlapack_check(A.layout == Layout::ColMajor ||
                      A.layout == Layout::RowMajor);
        if (A.layout == Layout::ColMajor) {
            m = A.m * sizeof(T);
            n = A.n;
        }
        else {
            m = A.n * sizeof(T);
            n = A.m;
        }

        tlapack_check(m >= 0);
        tlapack_check(n >= 0);

        tlapack_check(ldim >= m);
        if (n <= 1 || m == 0) ldim = m;
    }

    // Getters:
    inline constexpr byte* data() const { return ptr; }
    inline constexpr idx_t getM() const { return m; }
    inline constexpr idx_t getN() const { return n; }
    inline constexpr idx_t getLdim() const { return ldim; }
    inline constexpr idx_t size() const { return m * n; }

    /** Checks if a workspace is contiguous.
     *
     * @note This is one of the reasons to keep the attributes protected.
     *  The objects of this class always maintain one of the states:
     *      1. (n > 1) && (m > 0).
     *      2. ((n <= 1) || (m == 0)) && (ldim == m).
     */
    inline constexpr bool isContiguous() const { return (ldim == m); }

    /// Checks if a workspace contains the workspace of size m-by-n
    inline constexpr bool contains(idx_t m, idx_t n) const
    {
        if (isContiguous())
            return (size() >= (m * n));
        else
            return ((this->m >= m * n && this->n >= 1) ||
                    (this->m >= m && this->n >= n) ||
                    (this->n >= m * n && this->m >= 1));
    }

    /** Returns a workspace that is obtained by removing m*n bytes from the
     * current one.
     *
     * @note If the starting workspace is not contiguous, then we require one
     * of the following is true:
     *
     *          this->m == m * n && this->n >= 1
     *          this->m == m     && this->n >= n
     *          this->n == n     && this->m >= m
     *          this->n == m * n && this->m >= 1
     *
     * This is required to prevent bad partitioning by the algorithms using
     * this functionality.
     *
     * @param m Number of rows to be extracted.
     * @param n Number of columns to be extracted.
     */
    inline constexpr Workspace extract(idx_t m, idx_t n) const
    {
        if (isContiguous()) {
            tlapack_check(size() >= (m * n));
            // contiguous space in memory
            return Workspace(ptr + (m * n), size() - (m * n));
        }
        else if (this->m == m * n) {
            tlapack_check(this->n >= 1);

            return (this->n <= 2)
                       ? Workspace(ptr + ldim, this->m,
                                   this->n - 1)  // contiguous space in memory
                       : Workspace(ptr + ldim, this->m, this->n - 1,
                                   ldim);  // non-contiguous space in memory
        }
        else if (this->m == m) {
            tlapack_check(this->n >= n);

            return (this->n <= n + 1)
                       ? Workspace(ptr + n * ldim, this->m,
                                   this->n - n)  // contiguous space in memory
                       : Workspace(ptr + n * ldim, this->m, this->n - n,
                                   ldim);  // non-contiguous space in memory
        }
        else if (this->n == n) {
            tlapack_check(this->m >= m);

            // non-contiguous space in memory
            return Workspace(ptr + m, this->m - m, this->n, ldim);
        }
        else {
            tlapack_check(this->n == m * n && this->m >= 1);

            // non-contiguous space in memory
            return Workspace(ptr + 1, this->m - 1, this->n, ldim);
        }
    }

   private:
    idx_t m, n;  ///< Sizes
    byte* ptr;   ///< Pointer to array in memory
    idx_t ldim;  ///< Leading dimension
};

/// @brief Output information in the workspace query
struct WorkInfo {
    size_t m = 0;               ///< Number of rows needed in the Workspace
    size_t n = 1;               ///< Number of columns needed in the Workspace
    bool isContiguous = false;  ///< True if the Workspace is contiguous

    /// Constructor using sizes
    inline constexpr WorkInfo(size_t m = 0, size_t n = 1) noexcept : m(m), n(n)
    {}

    /// Size needed in the Workspace
    inline constexpr size_t size() const noexcept { return m * n; }

    /**
     * @brief Set the current object to a state that
     *  fit its current sizes and the sizes of workinfo.
     *
     * If sizes don't match, use simple solution: require contiguous space in
     * memory.
     *
     * @param[in] workinfo Another specification of work sizes
     */
    void minMax(const WorkInfo& workinfo) noexcept
    {
        const size_t m1 = workinfo.m;
        const size_t n1 = workinfo.n;
        const size_t s1 = workinfo.size();
        const size_t s = size();

        // If one of the objects is contiguous, then the result is contiguous
        if (isContiguous || workinfo.isContiguous) {
            m = std::max(s, s1);
            n = 1;
            isContiguous = true;
        }
        // Check if the current sizes cover the sizes from workinfo
        else if (m >= m1 && n >= n1) {
            // Nothing to do
        }
        // Check if the sizes from workinfo cover the current sizes
        else if (m1 >= m && n1 >= n) {
            // Use the sizes from workinfo
            m = m1;
            n = n1;
        }
        // Otherwise, use contiguous space with the maximum size
        else {
            m = std::max(s, s1);
            n = 1;
            isContiguous = true;
        }
    }

    /**
     * @brief Sum two object by matching sizes.
     *
     * If sizes don't match, use simple solution: require contiguous space in
     * memory.
     *
     * @param workinfo The object to be added to *this.
     * @return constexpr WorkInfo& The modified workinfo.
     */
    constexpr WorkInfo& operator+=(const WorkInfo& workinfo) noexcept
    {
        const size_t m1 = workinfo.m;
        const size_t n1 = workinfo.n;
        const size_t s1 = workinfo.size();

        // If one of the objects is contiguous, then the result is contiguous
        if (isContiguous || workinfo.isContiguous) {
            m = size() + s1;
            n = 1;
            isContiguous = true;
        }
        // Else, if first dimension matches, update second dimension
        else if (m == m1) {
            n += n1;
        }
        // Else, if second dimension matches, update first dimension
        else if (n == n1) {
            m += m1;
        }
        else  // Sizes do not match. Simple solution: contiguous space in memory
        {
            m = size() + workinfo.size();
            n = 1;
            isContiguous = true;
        }
        return *this;
    }

    constexpr WorkInfo transpose() noexcept { return WorkInfo(n, m); }
};

}  // namespace tlapack

#endif  // TLAPACK_WORKSPACE_HH
