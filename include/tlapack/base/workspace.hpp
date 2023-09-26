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

namespace tlapack {

/// @brief Output information in the workspace query
struct WorkInfo {
    size_t m = 0;               ///< Number of rows needed in the Workspace
    size_t n = 1;               ///< Number of columns needed in the Workspace
    bool isContiguous = false;  ///< True if the Workspace is contiguous
    bool isVector;  ///< True if the Workspace is a vector at compile time

    /// Constructor using sizes
    constexpr WorkInfo(size_t m, size_t n) noexcept
        : m(m), n(n), isVector(false)
    {}
    constexpr WorkInfo(size_t s = 0) noexcept : m(s), isVector(true) {}

    /// Size needed in the Workspace
    constexpr size_t size() const noexcept { return m * n; }

    /**
     * @brief Set the current object to a state that
     *  fit its current sizes and the sizes of workinfo.
     *
     * If sizes don't match, use simple solution: require contiguous space in
     * memory.
     *
     * @param[in] workinfo Another specification of work sizes
     */
    constexpr void minMax(const WorkInfo& workinfo) noexcept
    {
        const size_t& m1 = workinfo.m;
        const size_t& n1 = workinfo.n;
        const size_t s1 = workinfo.size();
        const size_t s = size();

        if (s1 == 0) {
            // Do nothing
        }
        else if (s == 0) {
            // Copy workinfo
            m = m1;
            n = n1;
            isContiguous = workinfo.isContiguous;
            isVector = workinfo.isVector;
        }
        else {
            if (isContiguous || workinfo.isContiguous) {
                // If one of the objects is contiguous, then the result is
                // contiguous
                m = std::max(s, s1);
                n = 1;
                isContiguous = true;
            }
            else if ((m >= m1 && n >= n1) ||
                     (workinfo.isVector && (m >= n1 && n >= m1))) {
                // Current shape cover workinfo's shape, nothing to be done.
                // If workinfo represents a vector, then we also check if the
                // vector fits transposed. In such a case, nothing to be done.
            }
            else if ((m1 >= m && n1 >= n) ||
                     (isVector && (m1 >= n && n1 >= m))) {
                // Same as above, but for the opposite case. THerefore, use the
                // sizes from workinfo.
                m = m1;
                n = n1;
            }
            else {
                // Workspaces are incompatible, use simple solution: contiguous
                // space in memory
                m = std::max(s, s1);
                n = 1;
                isContiguous = true;
            }

            // Update isVector
            isVector = isVector && workinfo.isVector;
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
        const size_t& m1 = workinfo.m;
        const size_t& n1 = workinfo.n;
        const size_t s1 = workinfo.size();
        const size_t s = size();

        if (s1 == 0) {
            // Do nothing
        }
        else if (s == 0) {
            // Copy workinfo
            m = m1;
            n = n1;
            isContiguous = workinfo.isContiguous;
            isVector = workinfo.isVector;
        }
        else {
            if (isContiguous || workinfo.isContiguous) {
                // One of the objects is contiguous, then the result is
                // contiguous
                m = s + s1;
                n = 1;
                isContiguous = true;
            }
            else if (n == n1) {
                // Second dimension matches, update first dimension
                m += m1;
            }
            else if (m == m1) {
                // First dimension matches, update second dimension
                n += n1;
            }
            else if (workinfo.isVector && (n == m1)) {
                // Second dimension matches the first dimension of workinfo and
                // workinfo is a vector. Therefore, the result is *this with
                // one additional row.
                m += 1;
            }
            else if (isVector && (m == n1)) {
                // First dimension matches the second dimension of workinfo and
                // *this is a vector. Therefore, the result is workinfo with
                // one additional row.
                m = m1 + 1;
                n = n1;
            }
            else {
                // Sizes do not match. Simple solution: contiguous space in
                // memory
                m = s + s1;
                n = 1;
                isContiguous = true;
            }

            // Update isVector
            isVector = isVector && workinfo.isVector;
        }

        return *this;
    }

    constexpr WorkInfo transpose() const noexcept
    {
        if (size() == 0 || isVector)
            return *this;
        else
            return WorkInfo(n, m);
    }
};

}  // namespace tlapack

#endif  // TLAPACK_WORKSPACE_HH
