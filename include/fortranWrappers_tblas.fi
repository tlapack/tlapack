interface

subroutine c_saxpy( n, alpha, x, incx, y, incy ) &
bind(C, name="saxpy")
    use, intrinsic :: iso_c_binding
    implicit none

    integer(c_int64_t), value :: n, incx, incy
    real(c_float), value :: alpha
    type(c_ptr), value :: x, y
end subroutine

subroutine c_daxpy( n, alpha, x, incx, y, incy ) &
bind(C, name="daxpy")
    use, intrinsic :: iso_c_binding
    implicit none

    integer(c_int64_t), value :: n, incx, incy
    real(c_double), value :: alpha
    type(c_ptr), value :: x, y
end subroutine

end interface