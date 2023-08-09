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

    constexpr WorkInfo transpose() const noexcept { return WorkInfo(n, m); }
};

}  // namespace tlapack

#endif  // TLAPACK_WORKSPACE_HH
