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
