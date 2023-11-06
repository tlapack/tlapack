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

#if __cplusplus >= 202002L
    #include <span>
#else
namespace std {
template <typename T>
class span {
    T* ptr_;
    std::size_t len_;

   public:
    template <class It>
    constexpr span(It first, std::size_t count) : ptr_(&(*first)), len_(count)
    {}
    constexpr T& operator[](std::size_t idx) const { return ptr_[idx]; }

    constexpr std::size_t size() const noexcept { return len_; }
    constexpr T* data() const noexcept { return ptr_; }
};
}  // namespace std
#endif

#include "tlapack/base/arrayTraits.hpp"

namespace tlapack {

namespace traits {
    namespace internal {
        template <typename>
        struct is_std_vector : std::false_type {};

        template <typename T, typename A>
        struct is_std_vector<std::vector<T, A>> : std::true_type {};

        template <typename T>
        struct is_std_vector<std::span<T>> : std::true_type {};
    }  // namespace internal

    template <typename T>
    inline constexpr bool is_stdvector_type = internal::is_std_vector<T>::value;
}  // namespace traits

// -----------------------------------------------------------------------------
// blas functions to access std::vector properties

// Size
template <class T, class Allocator>
constexpr auto size(const std::vector<T, Allocator>& x) noexcept
{
    return x.size();
}
template <class T>
constexpr auto size(const std::span<T>& x) noexcept
{
    return x.size();
}

// -----------------------------------------------------------------------------
// blas functions to access std::vector block operations

// slice
template <class vec_t,
          class SliceSpec,
          std::enable_if_t<traits::is_stdvector_type<vec_t>, int> = 0>
constexpr auto slice(const vec_t& v, SliceSpec&& rows) noexcept
{
    assert((rows.first >= 0 && (std::size_t)rows.first < size(v)) ||
           rows.first == rows.second);
    assert(rows.second >= 0 && (std::size_t)rows.second <= size(v));
    assert(rows.first <= rows.second);

    using T = type_t<vec_t>;
    return std::span<T>((T*)v.data() + rows.first, rows.second - rows.first);
}

}  // namespace tlapack

#endif  // TLAPACK_STDVECTOR_HH
