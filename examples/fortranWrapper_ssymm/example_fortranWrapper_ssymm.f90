! Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
!
! This file is part of <T>LAPACK.
! <T>LAPACK is free software: you can redistribute it and/or modify it under
! the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

program example_fortranWrapper_ssymm
use constants
implicit none

    ! Constants
    character, parameter :: layout = 'C', side = 'L', uplo = 'U'
    integer(blas_size), parameter :: m = 100, n = 200, incx = 1e0
    integer(blas_size), parameter :: lda = m, ldb = m, ldc = m
    
    ! Arrays with zeros
    real(sp) :: A(m,m), B(m,n), C(m,n)

    ! Local variables
    integer(blas_size) :: i, j
    
    ! Error measurements
    real(sp) :: errorF

    ! External functions
    external :: ssymm
    real(sp), external :: snrm2
    intrinsic :: random_number, min
    
    ! Initialize A with random numbers
    call random_number( A )
    
    ! Put junk in the lower part of A
    do j = 1, m
        do i = j+1, m
            A(i,j) = -6.2598534e+18
        end do
    end do

    ! Set C using A
    do j = 1, min(m,n)
        do i = 1, j-1
            C(i,j) = A(i,j)
            C(j,i) = C(i,j)
        end do
        C(j,j) = A(j,j)
    end do

    ! Set the main diagonal of B with ones
    do i = 1, min(m,n)
        B(i,i) = 1.e0
    end do
    
    ! Call ssymm
    call ssymm( &
        layout, side, uplo, m, n, -1.e0, &
        A, lda, B, ldb, 1.e0, C, ldc )
    
    ! Call snrm2
    errorF = snrm2( m*n, C, incx )

    ! Output
    print *, "||C-AB||_F = ", errorF

end program