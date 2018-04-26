import numpy as np


def jp_free(karr, kf):
  """ Compton profile of the non-interacting Fermi gas
  example:

  pmags = np.linspace(1e-3, 2*kf, 25)
  jp0 = jp_free(pmags, kf)
  plt.plot(pmags, jp0)

  Args:
    karr (np.array): a list of k magnitudes to evaluate J0(p) on
    kf (float): magnitude of Fermi k vector
  Return
    np.array: J0(p, kf) evaluated at p=karr
  """
  # initialize Compton profile to zero
  jp0 = np.zeros(len(karr))

  # non-zero within Fermi surface
  norm = 3./(4*kf**3)
  sel = karr < kf
  jp0[sel] = norm*(kf**2-karr[sel]**2)
  return jp0


def compton_sum(pvec, kvecs, nkm, eps=5e-2):
  """ perform Compton integral without normalization
  cint(p) = \int d\vec{k} \delta(p-\vec{k}\cdot\hat{p}) n(\vec{k})

  The normalization in 3D is 2./density* 1/(2pi)^3, where
  norm = density = kf^3/(3*pi^2) for unpolarized eletron gas.

  One must also provide the area of the 2D slice to fully normalize
  the Compton profile.

  J(p) = norm*cint(p)/area2d

  For momentum distribution taken in a cubic box with side length blat,
  area2d = blat**2.

  If you have n(k) represented as a 3D function, use compton_profile

  Args:
    pvec (np.array): 1D vector representing incident momentum
    kvecs (np.array): a list of k vectors on which n(k) is defined
    nkm (np.array): 1D array of floats, n(k) evaluated on kvecs
    eps (float, optional): width of approximate delta function,
    default is 5e-2 in Hatree atomic units
  Return:
    float: cint(p)
  """

  # impose delta function to select a 2D slice of n(k)
  pmag  = np.linalg.norm(pvec)
  kproj = np.dot(kvecs, pvec)/pmag
  ksel  = np.absolute(kproj-pmag) < eps

  # integrate 2D slice
  nkvals = nkm[ksel]
  npts = len(nkvals)
  if npts == 0:
    return np.nan
  return nkvals.sum()/npts


def compton_profile(pvecs, fnk, blat, nx=64, **kwargs):
  """ perform Compton profile integral scaled by 0.5*density
  density/2* J(p) = 1/(2pi)^3*
    \int d\vec{k} \delta(p-\vec{k}\cdot\hat{p}) n(\vec{k})
  by sampling n(k) on a cubic grid with side length blat

  blat must be large enough to contain the non-zero parts of n(k)
  i.e. blat > 2*kf + blat/nx for ideal n(k)

  expect the user to multiply return values by 2./density to finish
  normalization. This avoids having to pass density in as a call parameter.

  example:
  pmags = np.linspace(1e-3,2*kf,25)
  pvecs100 = [[pmag, 0, 0] for pmag in pmags]

  jp = intnorm* int_nofk.compton_profile(pvecs100, fnk0, 2.1*kf)

  Args:
    pvecs (np.array): a list of momentum vectors to evaluate J(p) on
    fnk (function): momentum distribution n(k)
     e.g. n( [[0.1*kf,0,0],[1.1*kf,0,0]] ) = [1.0, 0.0] for free gas
    blat (float): cubic box side length in reciprocal space
     must be large enough to contain the non-zero parts of n(k)
    nx (int, optional): number of grid points in each dimension, default 32
    kwargs (dict): keyword arguments to pass down to compton_sum, {'eps':0.05}
  Return:
    np.array: J(p) at pvecs
  """
  # define integration grid
  import chiesa_correction as chc
  kgrid = chc.remove_com( blat/nx*chc.cubic_pos(nx) )

  # sample momentum distribution on grid
  nkm = fnk(kgrid)

  # perform Compton integral
  jp = np.array([compton_sum(pvec, kgrid, nkm, **kwargs) for pvec in pvecs])
  intnorm = 1./(2*np.pi)**3 * blat**2
  return intnorm*jp
