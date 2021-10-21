// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_MDSPAN_HH__
#define __TBLAS_MDSPAN_HH__

#include <cassert>              // For the assert macro
#include <experimental/mdspan>  // Use mdspan for multidimensional arrays

namespace blas {

using std::experimental::mdspan;

// -----------------------------------------------------------------------------
// Data traits for mdspan

// Data type
template< class ET, class Exts, class LP, class AP >
struct type_trait< mdspan<ET,Exts,LP,AP> > {
    using type = ET;
};
// Size type
template< class ET, class Exts, class LP, class AP >
struct sizet_trait< mdspan<ET,Exts,LP,AP> > {
    using type = typename mdspan<ET,Exts,LP,AP>::size_type;
};

// Size
template< class ET, class Exts, class LP, class AP >
inline constexpr auto
size( const mdspan<ET,Exts,LP,AP>& x ) {
    return x.size();
}
// Number of rows
template< class ET, class Exts, class LP, class AP >
inline constexpr auto
nrows( const mdspan<ET,Exts,LP,AP>& x ) {
    return x.extent(0);
}
// Number of columns
template< class ET, class Exts, class LP, class AP >
inline constexpr auto
ncols( const mdspan<ET,Exts,LP,AP>& x ) {
    return x.extent(1);
}

// Submatrix
template< class ET, class Exts, class LP, class AP,
          class SliceSpecRow, class SliceSpecCol >
constexpr auto submatrix(
    const mdspan<ET,Exts,LP,AP>& A,
    SliceSpecRow rows,
    SliceSpecCol cols ) noexcept {
    return std::experimental::submdspan( A, rows, cols );
}

namespace internal{

using std::experimental::dextents;
using std::experimental::layout_stride;

// // -----------------------------------------------------------------------------
// /// layout_colmajor Column Major layout for mdspan
// struct layout_colmajor {
//     template <class Extents>
//     struct mapping {
//         static_assert(Extents::rank() == 2, "layout_colmajor is a 2D layout");

//         // for convenience
//         using idx_t = typename Extents::size_type;

//         // constructor
//         mapping( const Extents& exts, idx_t ldim ) noexcept
//         : extents_(exts), ldim_(ldim) {}

//         // Default constructors
//         mapping() noexcept = default;
//         mapping(const mapping&) noexcept = default;
//         mapping(mapping&&) noexcept = default;
//         mapping& operator=(mapping const&) noexcept = default;
//         mapping& operator=(mapping&&) noexcept = default;
//         ~mapping() noexcept = default;

//         //------------------------------------------------------------
//         // Required members

//         constexpr idx_t
//         operator()(idx_t row, idx_t col) const noexcept {
//             return row + col * ldim_;
//         }

//         constexpr idx_t
//         required_span_size() const noexcept {
//             return extents_.extent(1) * ldim_;
//         }

//         // Mapping is always unique
//         static constexpr bool is_always_unique() noexcept { return true; }
//         constexpr bool is_unique() const noexcept { return true; }

//         // Only contiguous if extents_.extent(0) == ldim_
//         static constexpr bool is_always_contiguous() noexcept { return false; }
//         constexpr bool is_contiguous() const noexcept { return (extents_.extent(0) == ldim_); }

//         // Mapping is always strided: strides: (1,ldim_)
//         static constexpr bool is_always_strided() noexcept { return true; }
//         constexpr bool is_strided() const noexcept { return true; }

//         // Get the extents
//         inline constexpr Extents extents() const noexcept { return extents_; };

//         // Get the leading dimension
//         inline constexpr idx_t ldim() const noexcept { return ldim_; };

//         private:
//             Extents extents_;
//             idx_t ldim_; // leading dimension
//     };
// };

// -----------------------------------------------------------------------------
/** Returns a Matrix object representing a column major matrix
 * 
 * @param A                 serial data
 * @param m                 number of rows
 * @param n                 number of columns
 * @param lda               leading dimension 
 * 
 * @return mdspan< T, dextents<2>, layout_stride > 
 *      matrix object using the abstraction A(i,j) = i + j * lda
 */
template< typename T >
inline auto colmajor_matrix(
    T* A, 
    dextents<2>::size_type m, 
    dextents<2>::size_type n, 
    dextents<2>::size_type lda )
{
    using extents = dextents<2>;
    using strides = dextents<2>;
    using mapping = typename layout_stride::template mapping<extents>;

    return mdspan< T, extents, layout_stride > (
        A, mapping( extents(m,n), strides(1,lda) )
    );
}

template< typename T >
inline auto vector(
    T* x,
    dextents<1>::size_type n,
    dextents<1>::size_type ldim )
{
    using extents = dextents<1>;
    using strides = dextents<1>;
    using mapping = typename layout_stride::template mapping<extents>;

    return mdspan< T, extents, layout_stride > (
        x, mapping( extents(n), strides(ldim) )
    );
}

} // namespace internal

} // namespace blas

#endif // __TBLAS_MDSPAN_HH__