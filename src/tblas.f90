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

    subroutine ssymm ( &
        layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc )
    implicit none
        integer, parameter :: wp = sp
    
        character          :: layout, side, uplo
        integer(blas_size) :: m, n, lda, ldb, ldc
        real(wp)           :: alpha, beta
        real(wp), target   :: A, B, C
    
        character(c_char) :: c_layout, c_side, c_uplo
        c_layout = layout
        c_side = side
        c_uplo = uplo
    
        call ssymm_ ( &
            c_layout, c_side, c_uplo, m, n, alpha, &
            c_loc(A), lda, c_loc(B), ldb, beta, c_loc(C), ldc )
    end subroutine

end module
