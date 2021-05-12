module blas
    use, intrinsic :: iso_c_binding
    private
    public :: saxpy

    include "fortranWrappers_tblas.fi"

contains

    subroutine saxpy ( n, alpha, x, incx, y, incy )
    implicit none
        integer :: n, incx, incy
        real    :: alpha
        real, target :: x(*), y(*)

        call c_saxpy( int(n, c_int64_t), alpha, &
            c_loc(x), int(incx, c_int64_t), &
            c_loc(y), int(incy, c_int64_t) )
    end subroutine

end module
