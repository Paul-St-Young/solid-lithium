import yaml
import numpy as np
from qharv.plantation import sugar

# Fermi k vector magnitude of homogeneous electron gas (heg) at density rs
heg_kfermi = lambda rs:((9*np.pi)/(4.*rs**3.))**(1./3)

# ================= level 0: raw data =================


def unfold_inv(kvecs, nkm, nke):
  """ unfold inversion symmetry of n(k) data

  Args:
    kvecs (np.array): k vectors
    nkm   (np.array): n(k) mean
    nke   (np.array): n(k) error
  Return:
    tuple: (kvecs1, nkm1, nke1) unfolded data
  """

  kvecs1 = np.concatenate([kvecs,-kvecs], axis=0)
  nkm1 = np.concatenate([nkm]*2, axis=0)
  nke1 = np.concatenate([nke]*2, axis=0)

  return kvecs1, nkm1, nke1
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
