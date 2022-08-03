! Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
!
! This file is part of <T>LAPACK.
! <T>LAPACK is free software: you can redistribute it and/or modify it under
! the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

subroutine saxpy ( n, alpha, x, incx, y, incy )
use, intrinsic :: iso_c_binding
use constants, &
only: wp => sp, &
      blas_int, &
      blas_size
implicit none

    integer(blas_size) :: n
    integer(blas_int)  :: incx, incy
    real(wp)           :: alpha
    real(wp), target   :: x(*), y(*)

    include "tlapack.fi"

    call saxpy_( n, alpha, c_loc(x), incx, c_loc(y), incy )
end subroutine

subroutine daxpy ( n, alpha, x, incx, y, incy )
use, intrinsic :: iso_c_binding
use constants, &
only: wp => dp, &
      blas_int, &
      blas_size
implicit none

    integer(blas_size) :: n
    integer(blas_int)  :: incx, incy
    real(wp)           :: alpha
    real(wp), target   :: x(*), y(*)

    include "tlapack.fi"

    call daxpy_( n, alpha, c_loc(x), incx, c_loc(y), incy )
end subroutine

subroutine caxpy ( n, alpha, x, incx, y, incy )
use, intrinsic :: iso_c_binding
use constants, &
only: wp => sp, &
      blas_int, &
      blas_size
implicit none

    integer(blas_size) :: n
    integer(blas_int)  :: incx, incy
    complex(wp)           :: alpha
    complex(wp), target   :: x(*), y(*)

    include "tlapack.fi"

    call caxpy_( n, alpha, c_loc(x), incx, c_loc(y), incy )
end subroutine

subroutine zaxpy ( n, alpha, x, incx, y, incy )
use, intrinsic :: iso_c_binding
use constants, &
only: wp => dp, &
      blas_int, &
      blas_size
implicit none

    integer(blas_size) :: n
    integer(blas_int)  :: incx, incy
    complex(wp)           :: alpha
    complex(wp), target   :: x(*), y(*)

    include "tlapack.fi"

    call zaxpy_( n, alpha, c_loc(x), incx, c_loc(y), incy )
end subroutine
