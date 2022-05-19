 subroutine fortran_slaqr0(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi,&
    Z, ldz, info, n_aed, n_sweep, n_shifts) bind(c,name='fortran_slaqr0')
    use,intrinsic:: iso_c_binding, only: c_float, c_int, c_bool
    use, intrinsic :: iso_fortran_env
    implicit none
    integer(c_int)::n,ilo, ihi, ldh, ldz, info, n_aed, n_sweep, n_shifts
    real(c_float),dimension(*):: H, wr, wi, Z
    logical(c_bool) :: wantt, wantz
    external slaqr0

    real :: dummywork(1)
    real, allocatable ::work(:)
    integer :: lwork
 
    call slaqr0( wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, 1, n, Z, ldz, dummywork, -1, info,&
                 n_aed, n_sweep, n_shifts )
    lwork = int( dummywork(1) )
    allocate( work(lwork) )
    call slaqr0( wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, 1, n, Z, ldz, work, lwork, info,&
                 n_aed, n_sweep, n_shifts )
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
    real(c_float) :: H(ldh, *), sr(*), si(*), Z(ldz,*)
    logical(c_bool) :: wantt, wantz
    external slaqr0

    real :: dummywork(1)
    real, allocatable ::work(:)
    integer :: lwork, kv, kt, nho, kwv, nve

    KV = N - NW + 1
    KT = NW + 1
    NHO = ( N-NW-1 ) - KT + 1
    KWV = NW + 2
    NVE = ( N-NW ) - KWV + 1
 


    call slaqr2( wantt, wantz, n, ilo, ihi, nw, H, ldh, 1, n, Z, ldz, ns, nd, sr, si,&
                 H( KV, 1 ), LDH, NHO, H( KV, KT ), LDH, NVE, H( KWV, 1 ), LDH,&
                 dummywork, -1 )
    lwork = int( dummywork(1) )
    allocate( work(lwork) )
    call slaqr2( wantt, wantz, n, ilo, ihi, nw, H, ldh, 1, n, Z, ldz, ns, nd, sr, si,&
                 H( KV, 1 ), LDH, NHO, H( KV, KT ), LDH, NVE, H( KWV, 1 ), LDH,&
                 work, lwork )
    deallocate(work)

 end subroutine

 subroutine fortran_slaqr5(wantt, wantz, n, ilo, ihi, nshifts, sr, si, H, ldh, Z, ldz&
    ) bind(c,name='fortran_slaqr5')
    use,intrinsic:: iso_c_binding, only: c_float, c_int, c_bool
    use, intrinsic :: iso_fortran_env
    implicit none
    integer(c_int)::n,ilo, ihi, ldh, ldz, nshifts,&
         KDU, KU, KWH, NHO, KWV, NVE
    real(c_float) :: H(ldh, *), sr(*), si(*), Z(ldz,*)
    logical(c_bool) :: wantt, wantz
    external slaqr0

    real, allocatable ::V(:,:)

    KDU = 2*nshifts
    KU = N - KDU + 1
    KWH = KDU + 1
    NHO = ( N-KDU+1-4 ) - ( KDU+1 ) + 1
    KWV = KDU + 4
    NVE = N - KDU - KWV + 1

    allocate( V(3, nshifts) )

    call slaqr5( wantt, wantz, 1, n, ilo, ihi, nshifts, sr, si,&
                 H, ldh, 1, n, Z, ldz,&
                 V, 3, H( KU, 1 ), LDH, NVE, H( KWV, 1 ), LDH, NHO, H( KU, KWH ), LDH )

    deallocate( V )

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

 subroutine fortran_slanv2(a, b, c, d, rt1r, rt1i, rt2r, rt2i,&
    cs, sn) bind(c,name='fortran_slanv2')
    use,intrinsic:: iso_c_binding, only: c_float
    use, intrinsic :: iso_fortran_env
    implicit none
    real(c_float):: a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn
    external slanv2

    call slanv2( a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn )

 end subroutine