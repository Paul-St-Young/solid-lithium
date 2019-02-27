import yaml
import numpy as np
from qharv.plantation import sugar

# Fermi k vector magnitude of homogeneous electron gas (heg) at density rs
heg_kfermi = lambda rs:((9*np.pi)/(4.*rs**3.))**(1./3)

# ================= level 0: raw data =================

def get_nofk_h5(fh5):
  """ extract 3D n(k) stored in h5 file

  expected h5 file format:
    nofk.h5/twist000
      kvecs                    Dataset {3470, 3}
      nke                      Dataset {3470/8192}
      nkm                      Dataset {3470/8192}
    nofk.h5/twist001
      kvecs                    Dataset {3462, 3}
      nke                      Dataset {3462/8192}
      nkm                      Dataset {3462/8192}

  Args:
    fh5 (str): h5 file name
  Return:
    (np.array, np.array, np.array): (kvecs, nkm, nke)
  """
  from qharv.sieve.scalar_h5 import twist_concat_h5
  kvecs = twist_concat_h5(fh5, 'kvecs')
  nkm = twist_concat_h5(fh5, 'nkm')
  nke = twist_concat_h5(fh5, 'nke')
  return kvecs, nkm, nke

def get_full_nk(fh5):
  kvecs, nkm, nke = get_nofk_h5(fh5)
  return unfold_inv(kvecs, nkm, nke)

def unfold_inv(kvecs, nkm, nke):
  """ unfold inversion symmetry of n(k) data

  Args:
    kvecs (np.array): k vectors
    nkm   (np.array): n(k) mean
    nke   (np.array): n(k) error
  Return:
    tuple: (kvecs1, nkm1, nke1) unfolded data
  """
  # flip and concat
  kvecs1 = np.concatenate([kvecs, -kvecs], axis=0)
  nkm1 = np.concatenate([nkm]*2, axis=0)
  nke1 = np.concatenate([nke]*2, axis=0)
  # keep unique
  ukvecs, idx = np.unique(kvecs1, axis=0, return_index=True)
  return kvecs1[idx], nkm1[idx], nke1[idx]
# end def unfold_inv


def get_nofk(fjson):
  """ convenience function for extracting all n(k) data from fjson

  combine solith.li_nofk.qmc_nofk.nofk_all_twists with unfolded_inv

  Args:
    fjson (str): JSON file holding n(k) data
  Return:
    tuple: (kvecs, nkm, nke) momentum distribution with unfolded inv. symm.
  """
  from solith.li_nofk.qmc_nofk import nofk_all_twists
  import pandas as pd
  df = pd.read_json(fjson)
  sel = np.ones(len(df), dtype=bool)
  kvecs0, nkm0, nke0 = nofk_all_twists(df, sel)
  kvecs, nkm, nke = unfold_inv(kvecs0, nkm0, nke0)
  return kvecs, nkm, nke
# end def get_nofk

def get_knk(fh5, ymean='nkm'):
  import h5py
  fp = h5py.File(fh5, 'r')
  kvecs = fp['kvecs'][()]
  nkm = fp[ymean][()]
  fp.close()
  return kvecs, nkm

def get_knk_tgrid(fh5, ymean='nkm', keys=['tgrid', 'raxes', 'gvecs', 'nke']):
  import h5py
  data = {}
  fp = h5py.File(fh5, 'r')
  for key in keys:
    data[key] = fp[key][()]
  nkm = fp[ymean][()]
  fp.close()
  kvecs = np.dot(data['gvecs'], data['raxes']/data['tgrid'])
  return kvecs, nkm, data

# ================= level 1: isotropic fit =================


@sugar.check_file_before
def save_bspline(fyaml, tck):
  """ save Bspline coefficients to file, abort if file exists

  example:
    import scipy.interpolate as interp
    tck = interp.splrep(x, y)
    save_bspline('myspline.yml', tck)

  Args:
    fyaml (str): filename to store yaml entry
    tck (tuple): length-3 tuple storing (knots, coeffs, order)
  """
  knots, coeffs, order = tck
  entry = {
    'knots':knots.tolist()
   ,'coeffs':coeffs.tolist()
   ,'order':int(order)
  }
  with open(fyaml,'w') as f:
    yaml.dump(entry,f)
# end def save_bspline


def load_bspline(fyaml):
  """ inverse of save_bspline

  example:
    import scipy.interpolate as interp
    tck = load_bspline('myspline.yml')
    myy = interp.splev(x, tck)
    fit_err = abs(y - myy)

  Args:
    fyaml (str): filename holding yaml entry
  Return
    tuple: length-3 tuple storing (knots, coeffs, order)
  """
  with open(fyaml,'r') as f:
    entry = yaml.load(f)
  knots  = np.array(entry['knots'])
  coeffs = np.array(entry['coeffs'])
  order  = entry['order']

  tck = (knots, coeffs, order)
  return tck
# end def load_bspline


def step1d(kmags, kf, jump):
  stepy = np.zeros(len(kmags))
  sel = kmags < kf
  stepy[sel] = jump
  return stepy


@sugar.check_file_before
def save_iso_fit(nk_yml, kf, jump, tck):
  """ save isotropic fit to n(k)
  n(k) = jump * step1d(k-kf) + splev(k, tck)

  Args:
    nk_yml (str): yaml file to store fit
    kf (float): magnitude of Fermi k vector
    jump (float): renormalization factor at kf
    tck (tuple): (knots, coefficient, order) accepted by scipy.interpolate.spl
  """

  save_bspline(nk_yml, tck)
  with open(nk_yml, 'r') as f:
    entry = yaml.load(f)
    entry['kf'] = kf
    entry['jump'] = jump

  with open(nk_yml, 'w') as f:
    yaml.dump(entry, f)


def load_iso_fit(nk_yml):
  """ load isotropic fit to n(k)
  n(k) = jump * step1d(k-kf) + splev(k, tck)

  nk_yml must have [kf, jump, knots, coeffs, order]

  Args:
    nk_yml (str): yaml file dumped by save_iso_fit
  Return:
    tuple: (kf, jump, tck)
  """

  tck = load_bspline(nk_yml)
  with open(nk_yml, 'r') as f:
    entry = yaml.load(f)
    kf = entry['kf']
    jump = entry['jump']
  return kf, jump, tck

# Kulik G function
# ----
def rfunc(u):
  return 1-u*np.arctan(1./u)

def rfunc_der1(u):
  uinv = 1./u
  first = 1./(u+uinv)
  second = np.arctan(uinv)
  return first - second

def kulikg(x, fineu):
  Ru = rfunc(fineu)
  finey = x/np.sqrt(Ru)
  Ry = rfunc(finey)
  Ru_prime = rfunc_der1(fineu)
  term1 = Ru_prime/Ru
  term2 = fineu/(fineu+finey)
  term3 = (Ru-Ry)/(fineu-finey)
  integrand = term1*term2*term3
  val = np.trapz(integrand, fineu)
  return val

def eval_kulikg(finek, umin=1e-6, umax=200., nu=4096):
  """Evaluate Kulik's G function at selected k values

  Args:
    finek (np.array): 1D array of k values
  Return:
    np.array: finey, values of Kulik G function
  """
  # grid for 1D integration
  fineu = np.linspace(umin, umax, nu)
  finey = [kulikg(x, fineu) for x in finek]
  return np.array(finey)

def nk_near_kf(x, n1, A):
  arg = abs(1-x)
  return n1 + A*arg*np.log(arg)

def fit_nk_near_kf(myx, myym, myye, xmin, xmax, func=None, ax=None):
  if func is None:
    from solith.li_nofk.fit_nofk import nk_near_kf
    func = nk_near_kf
  from scipy.optimize import curve_fit
  sel = (xmin < myx) & (myx < xmax)
  popt, pcov = curve_fit(func, myx[sel], myym[sel],
    sigma=myye[sel], absolute_sigma=True)
  perr = np.sqrt(np.diag(pcov))
  if ax is not None:
    xmargin = 1e-4
    nx = 128
    line = ax.plot(myx[sel], myym[sel], 'o', fillstyle='none')
    plot_min = min(1-1e-4, xmin+xmargin)
    plot_max = max(1+1e-4, xmax+xmargin)
    finex = np.linspace(plot_min, plot_max, nx)
    ax.plot(finex, func(finex, *popt), c=line[0].get_color())
  return popt, perr
# ----

# ================= level 2: 2D fit =================


def hex2d(kxy, kf):
  """ 2D step function with hexagonal symmetry """
  # find projection direction (for half plane)
  kx, ky = kxy
  proj = np.zeros(len(kx))
  tan = ky/kx
  theta = np.arctan(tan)
  zone = np.floor((theta+np.pi/8)/(np.pi/4)).astype(int)
  theta1 = zone*np.pi/4

  # get projection
  proj = np.absolute(kx*np.cos(theta1)+ky*np.sin(theta1))

  # plug into 1d function
  myz = np.ones(kx.shape)
  sel = proj > kf
  myz[sel] = 0
  return myz
def hex_area(kf):
  return np.pi*kf**2


def disk2d(kxy, kf):
  """ 2D step function with cylindrical symmetry """
  kx, ky = kxy
  k = np.sqrt(kx*kx+ky*ky)
  z = np.ones(kx.shape)
  sel = k>kf
  z[sel]=0
  return z
def disk_area(kf):
  return 8*kf**2*np.tan(np.pi/8)


# ================= level 2: 1D slices =================
def slice1d(phat, kvecs, eps=1e-6):
  kp = np.einsum('ij,j->i', kvecs, phat)
  pmag = np.linalg.norm(phat, axis=-1)
  # projection of kvecs along phat
  kproj = kp[:, np.newaxis]*phat[np.newaxis, :]/pmag
  # projection perpendicular to phat
  kperp = kvecs - kproj
  kpmags = np.linalg.norm(kperp, axis=-1)
  sel = abs(kpmags)<eps
  return sel
