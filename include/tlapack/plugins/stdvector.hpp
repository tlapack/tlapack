/// @file stdvector.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STDVECTOR_HH
#define TLAPACK_STDVECTOR_HH

#include <vector>

#include "tlapack/LegacyVector.hpp"

namespace tlapack {

// -----------------------------------------------------------------------------
// blas functions to access std::vector properties

// Size
template <class T, class Allocator>
inline constexpr auto size(const std::vector<T, Allocator>& x) noexcept
{
    return x.size();
}

// -----------------------------------------------------------------------------
// blas functions to access std::vector block operations

// slice
template <class T, class Allocator, class SliceSpec>
inline constexpr auto slice(const std::vector<T, Allocator>& v,
                            SliceSpec&& rows) noexcept
{
    assert((rows.first >= 0 && (std::size_t)rows.first < size(v)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 && (std::size_t)rows.second <= size(v));
    assert(rows.first <= rows.second);

    return LegacyVector<T, std::size_t>(rows.second - rows.first,
                                        (T*)v.data() + rows.first);
}

}  // namespace tlapack

#endif  // TLAPACK_STDVECTOR_HH
