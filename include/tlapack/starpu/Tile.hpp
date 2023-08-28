/// @file starpu/Tile.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_TILE_HH
#define TLAPACK_STARPU_TILE_HH

#include <starpu.h>

#include "tlapack/starpu/filters.hpp"

namespace tlapack {
namespace starpu {

    /**
     * @brief Class for representing a tile of a matrix
     *
     * @details Objects of this class are used to represent tiles of a matrix.
     */
    struct Tile {
        const starpu_data_handle_t root_handle;  ///< Matrix tile handle
        const idx_t i, j;  ///< Tile starting indices (i,j)
        const idx_t m, n;  ///< Tile sizes (m,n)

        starpu_data_handle_t handle;  ///< Tile handle possibly partitioned
        bool partition_planned =
            false;  ///< True if there is a partition associated with this tile

        /**
         * @brief Construct a new Tile object using a Matrix tile handle and the
         * local partitioning information
         */
        Tile(starpu_data_handle_t tile_handle,
             idx_t i,
             idx_t j,
             idx_t m,
             idx_t n) noexcept
            : root_handle(tile_handle), i(i), j(j), m(m), n(n)
        {
            // Collect information about the tile
            const idx_t M = starpu_matrix_get_nx(tile_handle);
            const idx_t N = starpu_matrix_get_ny(tile_handle);

            // Partition the tile if it is not a full tile
            if (m != M || n != N) {
                idx_t pos[4] = {i, j, m, n};
                struct starpu_data_filter f_tile = {
                    .filter_func = filter_tile,
                    .nchildren = 1,
                    .filter_arg_ptr = (void*)pos};

                starpu_data_partition_plan(root_handle, &f_tile, &handle);
                partition_planned = true;
            }
            else {
                handle = root_handle;
            }

            assert(starpu_matrix_get_nx(handle) == m &&
                   starpu_matrix_get_ny(handle) == n && "Invalid tile size");
        }

        /// Destructor
        ~Tile() noexcept
        {
            if (partition_planned)
                starpu_data_partition_clean(root_handle, 1, &handle);
        }

        /**
         * @brief Create a compatible handles between two tiles
         *
         * When two tiles are used in the same task and one of them has WRITE
         * mode, they must be compatible. This means that:
         * 1. They are associated to different matrix tile handles
         * 2. They are associated to the same matrix tile handle, but they are
         * not overlapping.
         *
         * If case two happens, we must create a new partition of the matrix
         * tile handle so that StarPU tasks can be submitted.
         *
         * @note One must call clean_compatible_handles() after submitting the
         * desired task to clean the partition that was possibly created in this
         * routine.
         *
         * @param[out] handles Array of two handles to be used in the task
         *      On exit, handles[0] is the handle of the first tile and
         *      handles[1] is the handle of the second tile.
         * @param[in] A First tile
         * @param[in] B Second tile
         */
        static void create_compatible_handles(starpu_data_handle_t handles[2],
                                              const Tile& A,
                                              const Tile& B) noexcept
        {
            if (A.root_handle == B.root_handle) {
                idx_t pos[8] = {A.i, A.j, A.m, A.n, B.i, B.j, B.m, B.n};

                struct starpu_data_filter f_ntiles = {
                    .filter_func = filter_ntiles,
                    .nchildren = 2,
                    .filter_arg_ptr = (void*)pos};

                starpu_data_partition_plan(A.root_handle, &f_ntiles, handles);

                assert(starpu_matrix_get_nx(handles[0]) == A.m &&
                       starpu_matrix_get_ny(handles[0]) == A.n &&
                       "Invalid tile size");
                assert(starpu_matrix_get_nx(handles[1]) == B.m &&
                       starpu_matrix_get_ny(handles[1]) == B.n &&
                       "Invalid tile size");
            }
            else {
                handles[0] = A.handle;
                handles[1] = B.handle;
            }
        }

        /** Clean the partition created by create_compatible_handles()
         *
         * @param[in,out] handles Array of two handles to be used in the task
         *      On exit, the partition is cleaned if it was previously created
         *      by create_compatible_handles().
         * @param[in] A First tile
         * @param[in] B Second tile
         */
        static void clean_compatible_handles(starpu_data_handle_t handles[2],
                                             const Tile& A,
                                             const Tile& B) noexcept
        {
            if (A.root_handle == B.root_handle)
                starpu_data_partition_clean(A.root_handle, 2, handles);
        }

        /**
         * @brief Create a compatible handles between one output tile and two
         * input tiles.
         *
         * @see create_compatible_handles() for more details.
         *
         * @note One must call clean_compatible_inout_handles() after submitting
         * the task to clean the partition that was possibly created in this
         * routine.
         *
         * @param[out] handles Array of three handles to be used in the task.
         *      On exit, handles[0] is the handle of the output tile, handles[1]
         *      is the handle of the first input tile and handles[2] is the
         *      handle of the second input tile.
         * @param[in] A First input tile.
         * @param[in] B Second input tile.
         */
        void create_compatible_inout_handles(starpu_data_handle_t handles[3],
                                             const Tile& A,
                                             const Tile& B) const noexcept
        {
            if (root_handle == A.root_handle) {
                if (root_handle == B.root_handle) {
                    idx_t pos[12] = {i,   j,   m,   n,   A.i, A.j,
                                     A.m, A.n, B.i, B.j, B.m, B.n};

                    struct starpu_data_filter f_ntiles = {
                        .filter_func = filter_ntiles,
                        .nchildren = 3,
                        .filter_arg_ptr = (void*)pos};

                    starpu_data_partition_plan(root_handle, &f_ntiles, handles);

                    assert(starpu_matrix_get_nx(handles[0]) == m &&
                           starpu_matrix_get_ny(handles[0]) == n &&
                           "Invalid tile size");
                    assert(starpu_matrix_get_nx(handles[1]) == A.m &&
                           starpu_matrix_get_ny(handles[1]) == A.n &&
                           "Invalid tile size");
                    assert(starpu_matrix_get_nx(handles[2]) == B.m &&
                           starpu_matrix_get_ny(handles[2]) == B.n &&
                           "Invalid tile size");
                }
                else {
                    create_compatible_handles(handles, *this, A);
                    handles[2] = B.handle;
                }
            }
            else if (root_handle == B.root_handle) {
                create_compatible_handles(handles, *this, B);
                handles[2] = handles[1];
                handles[1] = A.handle;
            }
            else {
                handles[0] = handle;
                handles[1] = A.handle;
                handles[2] = B.handle;
            }
        }

        /** Clean the partition created by create_compatible_inout_handles()
         *
         * @param[in,out] handles Array of three handles to be used in the task.
         *      On exit, the partition is cleaned if it was previously created
         *      by create_compatible_inout_handles().
         * @param[in] A First tile
         * @param[in] B Second tile
         */
        void clean_compatible_inout_handles(starpu_data_handle_t handles[3],
                                            const Tile& A,
                                            const Tile& B) const noexcept
        {
            if (root_handle == A.root_handle) {
                if (root_handle == B.root_handle)
                    starpu_data_partition_clean(root_handle, 3, handles);
                else
                    starpu_data_partition_clean(root_handle, 2, handles);
            }
            else if (root_handle == B.root_handle) {
                starpu_data_handle_t aux = handles[1];
                handles[1] = handles[2];
                starpu_data_partition_clean(root_handle, 2, handles);
                handles[2] = handles[1];
                handles[1] = aux;
            }
        }
    };

}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_TILE_HPP