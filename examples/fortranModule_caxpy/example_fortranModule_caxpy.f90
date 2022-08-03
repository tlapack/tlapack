! Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
!
! This file is part of <T>LAPACK.
! <T>LAPACK is free software: you can redistribute it and/or modify it under
! the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

program example_fortranModule_caxpy
use tlapack
implicit none
    logical, parameter            :: verbose = .false.

    integer(blas_size), parameter :: n = 10
    integer(blas_int),  parameter :: incx = 1, incy = 1
    complex(sp),        parameter :: c = ( 0.e0, -1.e0 )
    
    complex(sp)                   :: x(n), y(n)
    integer(blas_size)            :: i

    real(sp)                      :: error1
    complex(sp)                   :: aux

    intrinsic                     :: abs

    ! Fill arrays
    do i = 1, n
        x(i) = complex(i, sp) * c 
        y(i) = 10*complex(i, sp) * ( 0.e0, 1.e0 )
    end do

    if( verbose ) then
        ! Print arguments
        print *, "[IN] x = "
        do i = 1, n
            print *, x(i)
        end do
        print *, "[IN] y = "
        do i = 1, n
            print *, y(i)
        end do
        print *, "[IN] c = ", c
    end if
    
    ! Call caxpy
    call caxpy( n, c, x, incx, y, incy )

    if( verbose ) then
        ! Print result
        print *, "caxpy: c*x + y = "
        do i = 1, n
            print *, y(i)
        end do
    end if

    ! Error
    error1 = 0.e0
    do i = 1, n
        aux = complex(i, sp) * ( -1.e0, 10.e0 )
        error1 = error1 + abs( y(i) - aux )
    end do
    print *, "||y_exact - y_caxpy||_1 = ", error1

end program
