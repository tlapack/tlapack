! Copyright (c) 2021, University of Colorado Denver. All rights reserved.
!
! This file is part of T-LAPACK.
! T-LAPACK is free software: you can redistribute it and/or modify it under
! the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

program example_fortranModule_saxpy
use blas
implicit none

    integer, parameter :: n = 10, incx = 1, incy = 1
    real, parameter :: alpha = 2.e0
    real :: x(n), y(n)

    integer :: i

    ! Fill arrays
    do i = 1, n
        x(i) = i
        y(i) = 10*i
    end do

    ! Print arguments
    print *, "[IN] x = "
    do i = 1, n
        print *, x(i)
    end do
    print *, "[IN] y = "
    do i = 1, n
        print *, y(i)
    end do
    print *, "[IN] alpha = ", alpha
    
    ! Call saxpy
    call saxpy( n, alpha, x, incx, y, incy )

    ! Print result
    print *, "saxpy: alpha*x + y = "
    do i = 1, n
        print *, y(i)
    end do

end program