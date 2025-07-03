/// @file lasrt.hpp
/// @author Brian Dang, University of Colorado Denver, USA
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlasrt.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LASRT_HH
#define TLAPACK_LASRT_HH

#include <algorithm>
#include <utility>
#include <vector>

#include "tlapack/base/utils.hpp"

namespace tlapack {

template <class idx_t, class d_t>
int lasrt(char id, idx_t n, d_t& d)
{
    const idx_t SELECT = 20;
    int info = 0;
    int dir = -1;

    if (id == 'd' || id == 'D')
        dir = 0;  // descending
    else if (id == 'i' || id == 'I')
        dir = 1;  // ascending

    if (dir == -1) return -1;
    if (n < 0) return -2;
    if (n <= 1) return 0;

    using T = real_type<decltype(d[0])>;
    std::vector<std::pair<idx_t, idx_t>> stack;
    stack.emplace_back(0, n - 1);

    while (!stack.empty()) {
        auto [start, end] = stack.back();
        stack.pop_back();

        if (end - start <= SELECT && end > start) {
            // Insertion sort
            for (idx_t i = start + 1; i <= end; ++i) {
                for (idx_t j = i; j > start; --j) {
                    if ((dir == 0 && d[j] > d[j - 1]) ||
                        (dir == 1 && d[j] < d[j - 1])) {
                        std::swap(d[j], d[j - 1]);
                    }
                    else {
                        break;
                    }
                }
            }
        }
        else if (end - start > SELECT) {
            // Median-of-three pivot
            int d1 = d[start];
            int d2 = d[end];
            idx_t mid = (start + end) / 2;
            int d3 = d[mid];
            int pivot;

            if (d1 < d2) {
                if (d3 < d1)
                    pivot = d1;
                else if (d3 < d2)
                    pivot = d3;
                else
                    pivot = d2;
            }
            else {
                if (d3 < d2)
                    pivot = d2;
                else if (d3 < d1)
                    pivot = d3;
                else
                    pivot = d1;
            }

            idx_t i = start;
            idx_t j = end;
            while (true) {
                while (i <= end && ((dir == 0 && d[i] > pivot) ||
                                    (dir == 1 && d[i] < pivot)))
                    ++i;
                while (j >= start && ((dir == 0 && d[j] < pivot) ||
                                      (dir == 1 && d[j] > pivot)))
                    --j;

                if (i < j) {
                    std::swap(d[i], d[j]);
                    ++i;
                    --j;
                }
                else {
                    break;
                }
            }

            if (j - start > end - j - 1) {
                stack.emplace_back(start, j);
                stack.emplace_back(j + 1, end);
            }
            else {
                stack.emplace_back(j + 1, end);
                stack.emplace_back(start, j);
            }
        }
    }

    return info;
}

}  // namespace tlapack

#endif  // TLAPACK_LASRT_HH
