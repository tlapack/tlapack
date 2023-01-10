!> @file nrm2.f90 
!! @author Weslley S Pereira, University of Colorado Denver, USA
!
! Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
!
! This file is part of <T>LAPACK.
! <T>LAPACK is free software: you can redistribute it and/or modify it under
! the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

function snrm2 ( n, x, incx )
use, intrinsic :: iso_c_binding
use constants, &
only: wp => sp, &
      blas_int, &
      blas_size
implicit none

    integer(blas_size) :: n
    integer(blas_int)  :: incx
    real(wp), target   :: x(*)
    real(wp)           :: snrm2

    include "tlapack.fi"

    snrm2 = snrm2_( n, c_loc(x), incx )
end function
