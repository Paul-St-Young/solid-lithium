import numpy as np

# ================= 1D =================


def jp1d(pmag, kmags, nkm, rs):
  """ compute one point on the Compton profile J(p) using 1D n(k)

  Args:
    pmag (float): momentum value p in J(p)
    kmags (np.array): momentum magnitudes where n(k) is available
    nkm (np.array): n(k) mean (no error)
    rs (float): electron density measured by Wigner-Seitz radius
  Return:
    float: J(p)
  """
  from solith.li_nofk.fit_nofk import heg_kfermi
  kf = heg_kfermi(rs)
  norm = 3./(4*kf**3)
  sel = kmags>pmag
  intval = np.trapz(nkm[sel]*kmags[sel], x=kmags[sel])
  return intval*norm


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


# ================= 2D =================


def pvec_slice(pvec, kvecs, eps):
  """ get a 2D slice of 3D n(k) near plane perpendicular to pvec
  Args:
    pvec (np.array): vector defining cutting plane
    eps (float): include all points with distance < eps from plane
  Return:
    np.array: sel, boolean selector array to pass into kvecs
  """
  pmag  = np.linalg.norm(pvec)
  kproj = np.dot(kvecs, pvec)/pmag
  ksel  = np.absolute(kproj-pmag) < eps
  return ksel


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
  ksel = pvec_slice(pvec, kvecs, nkm, eps)

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
  kgrid = chc.remove_com(blat/nx*chc.cubic_pos(nx))

  # sample momentum distribution on grid
  nkm = fnk(kgrid)

  # perform Compton integral
  jp = np.array([compton_sum(pvec, kgrid, nkm, **kwargs) for pvec in pvecs])
  intnorm = 1./(2*np.pi)**3 * blat**2
  return intnorm*jp


def get_kxy(kvecs1, direction='100'):
  """ map 3D kvectors on to 2D kvectors for the given direction

  !!!! assume kvecs1 lie on the same plane along direction

  Args:
    kvecs1 (np.array): 3D k vectors
    direction (str): one of '100', '110', '111'
  Return:
    kxy (np.array): kx, ky = kxy.T
  """
  if direction == '100':
    # x-y mapping along [100] direction
    kxy = np.zeros([len(kvecs1), 2])
    kxy[:, 0] = -kvecs1[:, 2]
    kxy[:, 1] = kvecs1[:, 1]
  elif direction == '110':
    kxy = np.zeros([len(kvecs1), 2])
    kxy[:, 0] = -kvecs1[:, 2]
    kxy[:, 1] = -kvecs1[:, 0]/np.sqrt(2)+kvecs1[:, 1]/np.sqrt(2)
  else:
    raise NotImplementedError()
  return kxy


def interpolate(kxy, nkm, finex, finey):
  from scipy.interpolate import griddata
  finexy = [[(x, y) for y in finey] for x in finex]
  finez = griddata(kxy, nkm, finexy, fill_value=0.0)
  return finez

# ================= 3D =================
from qharv.plantation import sugar
@sugar.skip_exist_file
def save_nk3d(fh5, kvecs, nkm):
  from qharv.reel.config_h5 import saveh5
  data = np.zeros([len(nkm), 4])
  data[:, :3] = kvecs
  data[:,  3] = nkm
  saveh5(fh5, data)

def load_nk3d(fh5, name):
  import h5py
  fp = h5py.File(fh5, 'r')
  data = fp[name].value
  fp.close()

  kvecs = data[:, :3]
  nkm = data[:, 3]
  return kvecs, nkm
