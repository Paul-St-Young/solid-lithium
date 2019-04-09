import numpy as np

# ================= 1D =================

def jp_no_norm(p, uk, unkm):
  """Integrate isotropic k*n(k) in 3D"""
  sel = uk>=p
  return 2*np.pi*np.trapz(uk[sel]*unkm[sel], uk[sel])

def calc_jp1d(uk, unkm):
  """Calculate J(p) from n(k)"""
  jp = [jp_no_norm(p, uk, unkm) for p in uk]
  return np.array(jp)

def calc_nk1d(up, ujpm):
  """Calculate n(k) from J(p)"""
  djp = np.diff(ujpm)/np.diff(up)
  upm = 0.5*(up[1:]+up[:-1])
  norm = -1./(2*np.pi*upm)
  return upm, norm*djp

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
def get_phat(direction):
  """Get the unit vector along given direction

  Args:
    direction (str): Mi
  """
  if direction == '100':
    phat = np.array([1., 0, 0])
  elif direction == '110':
    phat = np.array([1., 1., 0])/2**0.5
  elif direction == '111':
    phat = np.array([1., 1., 1.])/3**0.5
  else:
    raise RuntimeError('unknown direction %s' % direction)
  return phat

def slice2d_sels(phat, kvecs, kmin, kmax, ndig=6):
  if not np.isclose(np.linalg.norm(phat), 1):
    raise RuntimeError('%s should be a unit vector' % str(phat))
  # get projected kmags along phat
  kpmags = np.einsum('ij,j->i', kvecs, phat)
  # find unique planes within given region
  pmags = np.unique(kpmags.round(ndig))
  psel = (kmin <= pmags) & (pmags <= kmax)
  upmags = pmags[psel]
  # construct selectors
  eps = 2*10**(-ndig)
  sels = []
  for pmag in upmags:
    sel = abs(kpmags-pmag) < eps
    sels.append(sel)
  return sels, upmags

def calc_jp2d(kvecs, nkm, direction='100', pmin=0, pmax=2., eps=1e-5, verbose=False):
  """Calculate Compton profile from 3D n(k) along one direction.
  !!!! Assume kvecs is a subset of a cubic regular grid.
  expect nkm to go from 0 to 2
  still need to divide by kvol =(2pi)^3/vol in 3D; for valence vol = 4pi rs^3/3

  Args:
    kvecs (np.array): (nk, ndim) kgrid for n(k)
    nkm (np.array): (nk,) 3D n(k)
    direction (str, optional): one of ['100', '110', '111'], default '100'
    pmin (float, optional): minimum momentum to calculate J(p)
    pmax (float, optional): maximum momentum to calculate J(p)
    eps (float, optional): tolerance for selecting plane, default 1e-5
    verbose (bool, optional): report progress if true
  Return:
    (np.array, np.array): (pmags, jpm), Compton profile
  """
  kx = np.unique(kvecs[:, 0])
  dk = kx[1]-kx[0]
  phat = get_phat(direction)
  kpmags = np.einsum('ij,j->i', kvecs, phat)
  pmags = np.unique(kpmags)
  psel = (pmin <= pmags) & (pmags <= pmax)
  upmags = pmags[psel]
  if verbose:
    from progressbar import ProgressBar
    bar = ProgressBar(maxval=len(upmags))
  jpl = []
  for ip, pmag in enumerate(upmags):
    # select plane
    sel = abs(kpmags-pmag) < eps
    if not sel.any():
      raise RuntimeError('failed to select plane %s' % pmag)
    # integrate plane
    nsum = nkm[sel].sum()
    jpl.append(nsum)
    if verbose:
      bar.update(ip)
  jpm = np.array(jpl)
  # normalize
  if direction == '100':
    norm = 1.
  elif direction == '110':
    norm = 2**0.5
  elif direction == '111':
    norm = 3**0.5
  else:
    raise RuntimeError('unknown direction %s' % direction)
  jpm *= dk**2*norm
  return upmags, jpm


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
  kx = kvecs1[:, 0]
  ky = kvecs1[:, 1]
  kz = kvecs1[:, 2]
  kxy = np.zeros([len(kvecs1), 2])
  if direction == '100':
    # x-y mapping along [100] direction
    kxy[:, 0] = ky
    kxy[:, 1] = kz
  elif direction == '110':
    kxy[:, 0] = (ky-kx)/np.sqrt(2)
    kxy[:, 1] = kz
  elif direction == '111':
    kxy[:, 0] = (kz-kx-ky)/np.sqrt(3)
    kxy[:, 1] =  (ky-kx)/np.sqrt(2)
  else:
    raise NotImplementedError()
  return kxy


def interpolate(kxy, nkm, finex, finey):
  from scipy.interpolate import griddata
  finexy = [[(x, y) for y in finey] for x in finex]
  finez = griddata(kxy, nkm, finexy, fill_value=0.0)
  return finez

def show_pcmesh(ax, kxy, nkm, kmax, nx, **kwargs):
  """ Show 2D n(k) on a square (-kmax, kmax) with nx points each dimension

  Args:
    ax (plt.Axes): matplotlib Axes object
    kxy (np.array): kvectors of shape (nk, ndim=2)
    nkm (np.array): n(k) mean values shape (nk,)
    kmax (float): max k along x
    nx (int): number of points along x
    kwargs (dict, optional): keyword arguments to pcolormesh
  Return:
    plt.QuadMesh: pcolormesh
  """
  # interpolate on meshgrid
  finex = np.linspace(-kmax, kmax, nx)
  finey = finex
  finez = interpolate(kxy, nkm, finex, finey)
  # show mesh
  xx, yy = np.meshgrid(finex, finey)
  qm = ax.pcolormesh(xx, yy, finez.reshape(nx, nx), **kwargs)
  return qm

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
