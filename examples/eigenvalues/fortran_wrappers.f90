subroutine fortran_slaqr0(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi,&
    Z, ldz, info) bind(c,name='fortran_slaqr0')
    use,intrinsic:: iso_c_binding, only: c_float, c_int, c_bool
    use, intrinsic :: iso_fortran_env
    implicit none
    integer(c_int)::n,ilo, ihi, ldh, ldz, info
    real(c_float),dimension(*):: H, wr, wi, Z
    logical(c_bool) :: wantt, wantz
    external slaqr0

    real :: dummywork(1)
    real, allocatable ::work(:)
    integer :: lwork
 
    call slaqr0( wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, 1, n, Z, ldz, dummywork, -1, info )
    lwork = int( dummywork(1) )
    allocate( work(lwork) )
    call slaqr0( wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, 1, n, Z, ldz, work, lwork, info )
    deallocate(work)

 end subroutine

 subroutine fortran_slahqr(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi,&
    Z, ldz, info) bind(c,name='fortran_slahqr')
    use,intrinsic:: iso_c_binding, only: c_float, c_int, c_bool
    use, intrinsic :: iso_fortran_env
    implicit none
    integer(c_int)::n,ilo, ihi, ldh, ldz, info
    real(c_float),dimension(*):: H, wr, wi, Z
    logical(c_bool) :: wantt, wantz
    external slahqr
 
    call slahqr( wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, 1, n, Z, ldz, info )

 end subroutine

 subroutine fortran_slaqr2(wantt, wantz, n, ilo, ihi, nw, H, ldh, Z, ldz,&
    ns, nd, sr, si) bind(c,name='fortran_slaqr2')
    use,intrinsic:: iso_c_binding, only: c_float, c_int, c_bool
    use, intrinsic :: iso_fortran_env
    implicit none
    integer(c_int)::n,ilo, ihi, ldh, ldz, nw, ns, nd
    real(c_float),dimension(*):: H, sr, si, Z
    logical(c_bool) :: wantt, wantz
    external slaqr0

    real :: dummywork(1)
    real, allocatable ::work(:),V(:,:), T(:,:), WV(:,:)
    integer :: lwork

    allocate( V(nw, nw) )
    allocate( T(nw, nw) )
    allocate( WV( n, nw ) )
 


    call slaqr2( wantt, wantz, n, ilo, ihi, nw, H, ldh, 1, n, Z, ldz, ns, nd, sr, si,&
                 V, nw, nw, T, nw, n, WV, n,  dummywork, -1 )
    lwork = int( dummywork(1) )
    allocate( work(lwork) )
    call slaqr2( wantt, wantz, n, ilo, ihi, nw, H, ldh, 1, n, Z, ldz, ns, nd, sr, si,&
                 V, nw, nw, T, nw, n, WV, n, work, lwork )
    deallocate(work)

    deallocate( V, T, WV )

 end subroutine

 subroutine fortran_slaqr5(wantt, wantz, n, ilo, ihi, nshifts, sr, si, H, ldh, Z, ldz&
    ) bind(c,name='fortran_slaqr5')
    use,intrinsic:: iso_c_binding, only: c_float, c_int, c_bool
    use, intrinsic :: iso_fortran_env
    implicit none
    integer(c_int)::n,ilo, ihi, ldh, ldz, nshifts, iblock
    real(c_float),dimension(*):: H, sr, si, Z
    logical(c_bool) :: wantt, wantz
    external slaqr0

    real, allocatable ::V(:,:), U(:,:), WV(:,:), WH(:,:)

    iblock = 3*nshifts

    allocate( V(3, nshifts) )
    allocate( U(iblock, iblock) )
    allocate( WV( n, iblock ) )
    allocate( WH( iblock, n ) )

    call slaqr5( wantt, wantz, 1, n, ilo, ihi, nshifts, sr, si,&
                 H, ldh, 1, n, Z, ldz,&
                 V, 3, U, iblock, n, WV, n, n, WH, iblock )

    deallocate( V, U, WV, WH )

 end subroutine

 subroutine fortran_sgehrd(n, ilo, ihi, H, ldh, tau,&
    info) bind(c,name='fortran_sgehrd')
    use,intrinsic:: iso_c_binding, only: c_float, c_int, c_bool
    use, intrinsic :: iso_fortran_env
    implicit none
    integer(c_int)::n,ilo, ihi, ldh, info
    real(c_float),dimension(*):: H, tau
    external slaqr0

    real :: dummywork(1)
    real, allocatable ::work(:)
    integer :: lwork

    call sgehrd( n, ilo, ihi, H, ldh, tau, dummywork, -1, info )
    lwork = int( dummywork(1) )
    allocate( work(lwork) )
    call sgehrd( n, ilo, ihi, H, ldh, tau, work, lwork, info )
    deallocate(work)

 end subroutine

 subroutine fortran_sorghr(n, ilo, ihi, H, ldh, tau,&
    info) bind(c,name='fortran_sorghr')
    use,intrinsic:: iso_c_binding, only: c_float, c_int, c_bool
    use, intrinsic :: iso_fortran_env
    implicit none
    integer(c_int)::n,ilo, ihi, ldh, info
    real(c_float),dimension(*):: H, tau
    external slaqr0

    real :: dummywork(1)
    real, allocatable ::work(:)
    integer :: lwork

    call sorghr( n, ilo, ihi, H, ldh, tau, dummywork, -1, info )
    lwork = int( dummywork(1) )
    allocate( work(lwork) )
    call sorghr( n, ilo, ihi, H, ldh, tau, work, lwork, info )
    deallocate(work)

 end subroutine
 