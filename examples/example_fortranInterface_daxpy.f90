! Copyright (c) 2021, University of Colorado Denver. All rights reserved.
!
! This file is part of T-LAPACK.
! T-LAPACK is free software: you can redistribute it and/or modify it under
! the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

program example_fortranInterface_daxpy
    use, intrinsic :: iso_c_binding

    include "cblas.fi"

    integer, parameter :: n = 10, incx = 1, incy = 1
    double precision, parameter :: alpha = 2.e0
    double precision, target :: x(n), y(n)

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
    
    ! Call daxpy
    call cblas_daxpy( int(n, c_int64_t), alpha, &
        c_loc(x), int(incx, c_int64_t), &
        c_loc(y), int(incy, c_int64_t) )

    ! Print result
    print *, "daxpy: alpha*x + y = "
    do i = 1, n
        print *, y(i)
    end do

end program