! Copyright (c) 2021, University of Colorado Denver. All rights reserved.
!
! This file is part of <T>LAPACK.
! <T>LAPACK is free software: you can redistribute it and/or modify it under
! the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

module tblas
    use, intrinsic :: iso_c_binding
    use constants
    public

    include "tblas.fi"

contains

    subroutine saxpy ( n, alpha, x, incx, y, incy )
    implicit none
        integer, parameter :: wp = sp

        integer(blas_size) :: n
        integer(blas_int)  :: incx, incy
        real(wp)           :: alpha
        real(wp), target   :: x(*), y(*)

        call saxpy_( n, alpha, c_loc(x), incx, c_loc(y), incy )
    end subroutine

    subroutine daxpy ( n, alpha, x, incx, y, incy )
    implicit none
        integer, parameter :: wp = dp

        integer(blas_size) :: n
        integer(blas_int)  :: incx, incy
        real(wp)           :: alpha
        real(wp), target   :: x(*), y(*)

        call daxpy_( n, alpha, c_loc(x), incx, c_loc(y), incy )
    end subroutine

    subroutine caxpy ( n, alpha, x, incx, y, incy )
    implicit none
        integer, parameter :: wp = sp

        integer(blas_size) :: n
        integer(blas_int)  :: incx, incy
        complex(wp)           :: alpha
        complex(wp), target   :: x(*), y(*)

        call caxpy_( n, alpha, c_loc(x), incx, c_loc(y), incy )
    end subroutine

    subroutine zaxpy ( n, alpha, x, incx, y, incy )
    implicit none
        integer, parameter :: wp = dp

        integer(blas_size) :: n
        integer(blas_int)  :: incx, incy
        complex(wp)           :: alpha
        complex(wp), target   :: x(*), y(*)

        call zaxpy_( n, alpha, c_loc(x), incx, c_loc(y), incy )
    end subroutine

end module
