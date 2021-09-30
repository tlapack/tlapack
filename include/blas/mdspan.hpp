// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_MDSPAN_HH__
#define __TBLAS_MDSPAN_HH__

#include <experimental/mdspan> // Use mdspan for multidimensional arrays

namespace blas {

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
// Dynamic matrix sizes
using matrix_extents = std::experimental::extents<
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent
>;

// -----------------------------------------------------------------------------
// Matrix mappings with dynamic extents
using StridedMapping  = typename std::experimental::layout_stride::template mapping<matrix_extents>;
using TiledMapping    = typename                    TiledLayout  ::template mapping<matrix_extents>;

// -----------------------------------------------------------------------------
// Column major matrix view with dynamic extents
template< typename T, typename Layout = std::experimental::layout_stride >
using Matrix = std::experimental::mdspan<
    T,
    matrix_extents,
    Layout
>;

} // namespace blas

#endif // __TBLAS_MDSPAN_HH__