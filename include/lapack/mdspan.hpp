// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_MDSPAN_HH__
#define __TLAPACK_MDSPAN_HH__

#include <experimental/mdspan> // Use mdspan for multidimensional arrays

namespace lapack {

// -----------------------------------------------------------------------------
/** TiledLayout Tiled layout for mdspan.
 * 
 * Column Major inter and intra tile organization 
 * 
 * For example, a serial array
 * 
 *   X = ... a b c d e f ... x y * * * * ...
 * 
 * using a tiled layout with
 *  number of rows    = 8
 *  number of columns = 7
 *  row_tile_size = 2
 *  col_tile_size = 3
 * would be represented as follows:
 *  __________________ _ _ 
 * |       |       |       !
 * |   1   |   5   |   9   !
 * |_______|_______|__ _ _ !
 * |       | a c e |       !
 * |   2   | b d f |   10  !
 * |_______|_______|__ _ _ !
 * |       |       | x * * !
 * |   3   |   7   | y * * !
 * |_______|_______|__ _ _ !
 * |       |       |       !
 * |   4   |   8   |   11  !
 * |_______|_______|__ _ _ !
 * 
 * where * represents data out of range.
 * 
 */
struct TiledLayout {
    template <class Extents>
    struct mapping {
        static_assert(Extents::rank() == 2, "TiledLayout is a 2D layout");

        // for convenience
        using size_type = typename Extents::size_type;

        // constructor
        mapping(
            const Extents& exts,    // matrix sizes
            size_type row_tile,
            size_type col_tile
        ) noexcept
            : extents_(exts)
            , row_tile_size_(row_tile)
            , col_tile_size_(col_tile)
        {} // Mind that it does not check for invalid values here.

        // Default constructors
        mapping() noexcept = default;
        mapping(const mapping&) noexcept = default;
        mapping(mapping&&) noexcept = default;
        mapping& operator=(mapping const&) noexcept = default;
        mapping& operator=(mapping&&) noexcept = default;
        ~mapping() noexcept = default;

        //------------------------------------------------------------
        // Helper members (not part of the layout concept)

        constexpr size_type
        n_row_tiles() const noexcept {
            return extents_.extent(0) / row_tile_size_ + size_type((extents_.extent(0) % row_tile_size_) != 0);
        }

        constexpr size_type
        n_column_tiles() const noexcept {
            return extents_.extent(1) / col_tile_size_ + size_type((extents_.extent(1) % col_tile_size_) != 0);
        }

        constexpr size_type
        tile_size() const noexcept {
            return row_tile_size_ * col_tile_size_;
        }

        size_type
        tile_offset(size_type row, size_type col) const noexcept {
            auto col_tile = col / col_tile_size_;
            auto row_tile = row / row_tile_size_;
            return (col_tile * n_row_tiles() + row_tile) * tile_size();
        }

        size_type
        offset_in_tile(size_type row, size_type col) const noexcept {
            auto t_row = row % row_tile_size_;
            auto t_col = col % col_tile_size_;
            return t_row + t_col * row_tile_size_;
        }

        //------------------------------------------------------------
        // Required members

        constexpr size_type
        operator()(size_type row, size_type col) const noexcept {
            return tile_offset(row, col) + offset_in_tile(row, col);
        }

        constexpr size_type
        required_span_size() const noexcept {
            return n_row_tiles() * n_column_tiles() * tile_size();
        }

        // Mapping is always unique
        static constexpr bool is_always_unique() noexcept { return true; }
        constexpr bool is_unique() const noexcept { return true; }

        // Only contiguous if extents fit exactly into tile sizes...
        static constexpr bool is_always_contiguous() noexcept { return false; }
        constexpr bool is_contiguous() const noexcept { 
            return (extents_.extent(0) % row_tile_size_ == 0) && (extents_.extent(1) % col_tile_size_ == 0);
        }

        // There is not always a regular stride between elements in a given dimension
        static constexpr bool is_always_strided() noexcept { return false; }
        constexpr bool is_strided() const noexcept { return false; }

        inline constexpr Extents
        extents() const noexcept {
            return extents_;
        };

        private:
            Extents extents_;
            size_type row_tile_size_; // row tile
            size_type col_tile_size_; // column tile
    };
};

// -----------------------------------------------------------------------------
/** Scaled accessor policy for mdspan.
 * @brief Allows for the lazy evaluation of the scale operation of arrays
 * 
 * @tparam ElementType  Type of the data
 * @tparam scalar_t     Type of the scalar multiplying the data
 */
template <class ElementType, class scalar_t>
struct scaled_accessor {
  
    using offset_policy = scaled_accessor;
    using element_type = ElementType;
    using pointer = ElementType*;
    using scalar_type = scalar_t;
    using scalar_ptr = scalar_t*;

    inline constexpr scaled_accessor() noexcept
    : scale_( 1 ) { };

    inline constexpr scaled_accessor( const scalar_t& scale ) noexcept
    : scale_( scale ) { };

    template< class OtherElementType, class otherScalar_t,
        enable_if_t< (
            is_convertible_v<
                typename scaled_accessor<OtherElementType, otherScalar_t>::pointer, 
                pointer >
            &&
            is_convertible_v< 
                typename scaled_accessor<OtherElementType, otherScalar_t>::scalar_ptr, 
                scalar_ptr >
        ), bool > = true
    >
    inline constexpr scaled_accessor( scaled_accessor<OtherElementType, otherScalar_t> ) noexcept {}

    inline constexpr pointer
    offset( pointer p, std::size_t i ) const noexcept {
        return p + i;
    }

    inline constexpr auto access(pointer p, std::size_t i) const noexcept {
        return scale_ * p[i];
    }

    inline constexpr scalar_type scale() const noexcept {
        return scale_;
    }

    private:
        scalar_t scale_;
};

// -----------------------------------------------------------------------------
// Dynamic matrix sizes
using matrix_extents = std::experimental::extents<
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent
>;

// -----------------------------------------------------------------------------
// Matrix mappings with dynamic extents
using StridedMapping  = typename layout_stride::template mapping<matrix_extents>;
using TiledMapping    = typename TiledLayout  ::template mapping<matrix_extents>;

// -----------------------------------------------------------------------------
// Column major matrix view with dynamic extents
template< typename T, typename Layout = layout_stride >
using Matrix = mdspan< T, matrix_extents, Layout >;

/**
 * @brief scale an array by a scalar alpha
 * 
 * It is a lazy evaluation strategy. The actual scale occurs element-wisely at the evaluation on each 
 * 
 * @tparam scalar_t Type of the scalar multiplying the data
 * @tparam array_t  Type of the array
 * @param alpha     Scalar
 * @param A         Array
 * @return An array   
 */
template<
    class scalar_t, class T, std::size_t... Exts, class LP, class AP,
    enable_if_t< is_same_v< AP, default_accessor<T> >
    , bool > = true
>
constexpr auto scale(
    const scalar_t& alpha, 
    const mdspan<T, std::experimental::extents<Exts...>, LP, AP>& A )
{
    using newAP = scaled_accessor<T,scalar_t>;
    return mdspan<T, std::experimental::extents<Exts...>, LP, newAP>(
        A.data(), A.mapping(), newAP( alpha )
    );
}

template<
    class scalar_t, class T, std::size_t... Exts, class LP, class AP, class otherScalar_t,
    enable_if_t< is_same_v< AP, scaled_accessor<T,otherScalar_t> >
    , bool > = true
>
constexpr auto scale(
    const scalar_t& alpha, 
    const mdspan<T, std::experimental::extents<Exts...>, LP, AP>& A )
{
    const auto beta = alpha * A.accessor().scale();
    using newAP = scaled_accessor<T,decltype(beta)>;
    return mdspan<T, std::experimental::extents<Exts...>, LP, newAP>(
        A.data(), A.mapping(), newAP( beta )
    );
}

template<
    class scalar_t, class T, std::size_t... Exts, class LP, class AP, class otherScalar_t,
    enable_if_t< is_same_v< AP, scaled_accessor<T,otherScalar_t> >
    , bool > = true
>
constexpr auto scale(
    const scalar_t& alpha, 
    const mdspan<T, std::experimental::extents<Exts...>, LP, AP>& A )
{
    const auto beta = alpha * A.accessor().scale();
    using newAP = scaled_accessor<T,decltype(beta)>;
    return mdspan<T, std::experimental::extents<Exts...>, LP, newAP>(
        A.data(), A.mapping(), newAP( beta )
    );
}


// -----------------------------------------------------------------------------
/** Returns a submatrix from the input matrix
 * 
 * The submatrix uses the same data from the matrix A.
 * 
 * It uses the function `std::experimental::submdspan` from https://github.com/kokkos/mdspan
 * Currently, it only allows for contiguous ranges
 * 
 * @param[in] A             Matrix
 * @param[in] rows          Rows used from A.
 *      It accepts values that can be converted to
 *          - a size_t integer: reprents the index of the row to be used.
 *          - a std::pair(a,b): uses rows from a to b-1
 *          - a std::experimental::full_extent: uses all rows
 * @param[in] cols          Columns used from A.
 *      It accepts values that can be converted to
 *          - a size_t integer: reprents the index of the column to be used.
 *          - a std::pair(a,b): uses columns from a to b-1
 *          - a std::experimental::full_extent: uses all columns
 * 
 * @return Matrix<T>        submatrix
 */
template< typename matrix_t, typename SliceSpecRow, typename SliceSpecCol >
constexpr auto submatrix( const matrix_t& A, SliceSpecRow rows, SliceSpecCol cols ) noexcept {
    return std::experimental::submdspan( A, rows, cols );
}

} // namespace lapack

#endif // __TLAPACK_MDSPAN_HH__