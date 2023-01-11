!> @file symm.f90 
!! @author Weslley S Pereira, University of Colorado Denver, USA
!
! Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
!
! This file is part of <T>LAPACK.
! <T>LAPACK is free software: you can redistribute it and/or modify it under
! the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

subroutine ssymm ( &
    layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc )
use, intrinsic :: iso_c_binding
use constants, &
only: wp => sp, &
      blas_size
implicit none

    character          :: layout, side, uplo
    integer(blas_size) :: m, n, lda, ldb, ldc
    real(wp)           :: alpha, beta
    real(wp), target   :: A, B, C

    character(c_char) :: c_layout, c_side, c_uplo

    include "tlapack.fi"

    c_layout = layout
    c_side = side
    c_uplo = uplo

    call ssymm_ ( &
        c_layout, c_side, c_uplo, m, n, alpha, &
        c_loc(A), lda, c_loc(B), ldb, beta, c_loc(C), ldc )
end subroutine
