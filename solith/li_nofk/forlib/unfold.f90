subroutine unfold_nofk(gvecs, nkm, mats, rgvecs, nkm1, filled, ndim, nk, nkr, ns)
  integer, intent(in) :: gvecs(nk, ndim), rgvecs(nkr, ndim)
  double precision, intent(in) :: mats(ns, ndim, ndim)
  double precision, intent(in) :: nkm(nk)
  double precision, intent(out) :: nkm1(nkr)
  logical, intent(out) :: filled(nkr)

  integer :: gmin(ndim), gmax(ndim), ng(ndim)
  double precision :: dg(ndim)
  integer :: gvec1(ndim), idx3d(ndim)
  integer idx1d
  logical outside

  gmin = minval(rgvecs, dim=1)
  gmax = maxval(rgvecs, dim=1)
  ng = gmax-gmin+1
  ! checked gmin, gmax, ng against chc.get_regular_grid_dimensions

  filled(:) = .false.
  do ig=1,nk
    do is=1,ns
      ! apply symmetry operation to gvec -> gvec1
      do j=1,ndim
        gvec1(j) = 0.0
        do i=1,ndim
          gvec1(j) = gvec1(j) + gvecs(ig, i)*mats(is, j, i)
        enddo
      enddo
      outside = .false.
      do i=1,ndim
        if ((gvec1(j).lt.gmin(j)).or.(gvec1(j).gt.gmax(j))) outside=.true.
      enddo
      if (outside) cycle

      idx3d = gvec1-gmin
      idx1d = idx3d(3) + idx3d(2)*ng(3) + idx3d(1)*ng(3)*ng(2) + 1
      if (filled(idx1d)) then
        cycle
      endif
      nkm1(idx1d) = nkm(ig)
      filled(idx1d) = .true.
    enddo
  enddo
end subroutine unfold_nofk
