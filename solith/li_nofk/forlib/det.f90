subroutine nofk(kvecs, cmat, evals, kmax, ecore, efermi, nbnd, npw, ndim, weights)
  double precision, intent(in) :: kvecs(npw, ndim)
  complex*16, intent(in) :: cmat(nbnd, npw)
  double precision, intent(in) :: evals(nbnd)
  double precision, intent(out) :: weights(npw)
  double precision, intent(in) :: kmax
  complex*16 pg2
  double precision kmags2(npw)
  double precision kmax2
  integer ib, ipw
  kmax2 = kmax**2
  weights(:) = 0
  do ipw=1, npw
    kmags2(ipw) = sum(kvecs(ipw, :)*kvecs(ipw, :))
  enddo
  do ib=1, nbnd
    if (evals(ib) .gt. efermi) then
      cycle
    endif
    if (evals(ib) .lt. ecore) then
      cycle
    endif
    do ipw=1, npw
      if (kmags2(ipw) .gt. kmax2) then
        cycle
      endif
      pg2 = conjg(cmat(ib, ipw))*cmat(ib, ipw)
      weights(ipw) = weights(ipw) + real(pg2)
    enddo
  enddo
end subroutine nofk

subroutine detsk_from_mij(mij, norb, sk0)
  complex*16, intent(in) :: mij(norb, norb)
  double precision, intent(out) :: sk0
  complex*16 sk_comp
  integer i, j
  sk_comp=0
  do i=1,norb
    do j=1,norb
      if (i.eq.j) cycle
      sk_comp = sk_comp+&
        mij(i,i)*conjg(mij(j,j))-mij(i,j)*conjg(mij(i,j))
    enddo
  enddo
  sk0 = 1.+real(sk_comp)/norb
end subroutine detsk_from_mij

subroutine calc_detsk(qvecs, raxes, gvecs, cmat,&
  nq, ndim, npw, norb,&
  sk0)
  integer, intent(in) :: qvecs(nq, ndim), gvecs(npw, ndim)
  double precision, intent(in) :: raxes(ndim, ndim)
  complex*16, intent(in) :: cmat(norb, npw)
  double precision, intent(out) :: sk0(nq)

  integer, allocatable :: ridx(:)
  integer :: qvec(ndim), gplusq(ndim), gmin(ndim), gmax(ndim)
  integer :: idx3d(ndim), gs(ndim)
  complex*16 :: mij(norb, norb)
  integer iq, ig, i, ngs
  logical outside
  complex*16 ci, cj
  double precision skval

  ! get regular grid dimensions
  gmin = minval(gvecs, dim=1)
  gmax = maxval(gvecs, dim=1)
  gs = gmax-gmin+1
  ngs = product(gs)

  ! map regular grid points to available data
  allocate(ridx(ngs))
  ridx(:) = -1
  do ig=1,npw
    idx3d = gvecs(ig, :)-gmin
    idx1d = idx3d(3) + idx3d(2)*gs(3) + idx3d(1)*gs(3)*gs(2) + 1
    ridx(idx1d) = ig
  end do

  do iq=1,nq
    ! step 1: calculate mij at q
    mij(:,:) = 0
    qvec = qvecs(iq, :)
    do ig=1,npw
      gplusq(:) = qvec(:)+gvecs(ig, :)
      ! look for g+q
      outside = .false.
      do i=1,ndim
        if ((gplusq(i).lt.gmin(i)).or.(gplusq(i).gt.gmax(i))) outside=.true.
      enddo
      if (outside) cycle
      idx3d = gplusq-gmin
      idx1d = idx3d(3) + idx3d(2)*gs(3) + idx3d(1)*gs(3)*gs(2) + 1
      if (ridx(idx1d).eq.-1) cycle
      do i=1,norb
        ci = cmat(i, ridx(idx1d))
        do j=1,norb
          cj = cmat(j, ig)
          mij(i,j) = mij(i,j) + conjg(ci)*cj
        enddo
      enddo
    enddo
    ! step 2: calculate sk0 using mij
    call detsk_from_mij(mij, norb, skval)
    sk0(iq) = skval
  enddo
  deallocate(ridx)
end subroutine calc_detsk
