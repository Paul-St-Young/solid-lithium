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

def get_one_nk3d(iconf, series, stat_dir):
  from qharv.reel import mole
  fregex = '*_conf%d_*.s%03d.nofk.h5' % (iconf, series)
  fh5 = mole.find(fregex, stat_dir)
  kvecs, nkm, nke = get_full_nk(fh5)
  return kvecs, nkm, nke

def get_pure_nk3d(iconf, dseries, stat_dir, vseries=0):
  kvecs0, nkm0, nke0 = get_one_nk3d(iconf, vseries, stat_dir)
  kvecs1, nkm1, nke1 = get_one_nk3d(iconf, dseries, stat_dir)
  assert np.allclose(kvecs0, kvecs1)
  kvecs2 = kvecs1
  nkm2 = 2*nkm1-nkm0
  nke2 = (4*nke1**2+nke0**2)**0.5
  return kvecs2, nkm2, nke2

def get_one_nk1d(iconf, series, stat_dir):
  from static_correlation import shavg
  kvecs, nkm, nke = get_one_nk3d(iconf, series, stat_dir)
  uk, unkm, unke = shavg(kvecs, nkm, nke)
  return uk, unkm, unke

def get_pure_nk1d(iconf, dseries, stat_dir, vseries=0):
  uk0, unkm0, unke0 = get_one_nk1d(iconf, vseries, stat_dir)
  uk1, unkm1, unke1 = get_one_nk1d(iconf, dseries, stat_dir)
  if not np.allclose(uk0, uk1):
    raise RuntimeError('VMC DMC kgrid mismatch')
  uk2 = uk1
  unkm2 = 2*unkm1-unkm0
  unke2 = (4*unke1**2+unke0**2)**0.5
  return uk2, unkm2, unke2

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


@sugar.skip_exist_file
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


@sugar.skip_exist_file
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
# ----

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

def fit_zkf(x, ym, ye,
  xmin, xleft, xright, xmax, **kwargs):
  popt1, perr1 = fit_nk_near_kf(x, ym, ye, xmin, xleft, **kwargs)
  popt2, perr2 = fit_nk_near_kf(x, ym, ye, xright, xmax, **kwargs)
  nleftm = popt1[0]
  nlefte = perr1[0]
  nrightm = popt2[0]
  nrighte = perr2[0]
  alm = popt1[1]
  ale = perr1[1]
  arm = popt2[1]
  are = perr2[1]
  zkfm = nleftm-nrightm
  zkfe = (nlefte**2+nrighte**2)**0.5
  data = {'nleft_mean': nleftm, 'nleft_error': nlefte,
          'nright_mean': nrightm, 'nright_error': nrighte,
          'aleft_mean': alm, 'aleft_error': ale,
          'aright_mean': arm, 'aright_error': are}
  return zkfm, zkfe, data

def get_loglog_left(myx0, myym0):
  # throw out first point (!!!! assume x=0)
  n0m = myym0[0]
  myx = myx0[1:]
  myym = n0m-myym0[1:]
  sel = (myx < 1.) & (myym > 0)
  return np.log10(myx[sel]), np.log10(myym[sel]), n0m

def get_loglog_right(myx0, myym0):
  sel = (myx0 > 1.) & (myym0 > 0)
  return np.log10(myx0[sel]), np.log10(myym0[sel])

def fit_loglog_oneside(log10x, log10y, xleft, xright, func=None, ax=None):
  from scipy.optimize import curve_fit
  if func is None:
    def linear(x, a, b):
      return a+b*x
    func = linear
  sel = (np.log10(xleft) < log10x) & (log10x < np.log10(xright))
  myx = log10x[sel]
  myy = log10y[sel]
  popt, pcov = curve_fit(func, myx, myy)
  perr = np.sqrt(np.diag(pcov))
  if ax is not None:
    line = ax.plot(myx, myy, 'o', fillstyle='none')
    xlim_left = min(xleft-.2, 1.0)
    xlim_right = max(xright+.2, 1.0)
    finex = np.linspace(np.log10(xlim_left), np.log10(xlim_right), 64)
    ax.plot(finex, func(finex, *popt), c=line[0].get_color())
  return popt, perr

def fit_loglog(myx, myym, xmin, xleft, xright, xmax, **kwargs):
  ax = kwargs.pop('ax', None)
  x1, y1, n0 = get_loglog_left(myx, myym)
  x2, y2 = get_loglog_right(myx, myym)
  if ax is not None:
    ax.plot(x1, y1, 'x')
    ax.plot(x2, y2, 'x')
  lpopt, lperr = fit_loglog_oneside(x1, y1, xmin, xleft, ax=ax)
  rpopt, rperr = fit_loglog_oneside(x2, y2, xright, xmax, ax=ax)
  entry = {'lpopt': lpopt, 'lperr': lperr, 'rpopt': rpopt, 'rperr': rperr}
  # interpret fits
  lalpha = 10**lpopt[0]
  lbeta = lpopt[1]
  ralpha = 10**rpopt[0]
  rbeta = rpopt[1]
  entry.update({
    'lalpha': lalpha, 'lbeta': lbeta, 'n0': n0,
    'ralpha': ralpha, 'rbeta': rbeta,
    'nleft': n0-lalpha, 'nright': ralpha, 'zeta': n0-lalpha-ralpha})
  return entry

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
  sel = k > kf
  z[sel] = 0
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
  sel = abs(kpmags) < eps
  return sel

def show_spline(ax, ux, uym, xleft, xright, smooth=0.0001, **kwargs):
  from scipy.interpolate import splrep, splev
  sel = (xleft<=ux) & (ux<=xright)
  nx = len(ux[sel])
  finex = np.linspace(xleft, xright, 10*nx)
  if nx < 4:  # fit a quadratic
    popt = np.polyfit(ux[sel], uym[sel], 2)
    finey = np.poly1d(popt)(finex)
  else:
    tck = splrep(ux[sel], uym[sel], s=smooth)
    finey = splev(finex, tck)
  line = ax.plot(finex, finey, **kwargs)
  return line

def show_data(ax, efunc, ux, uym, uye, xleft, xright, **kwargs):
  sel = (xleft<=ux) & (ux<=xright)
  line = efunc(ax, ux[sel], uym[sel], uye[sel], **kwargs)
  return line

def show_interp(ax, ux, uym, xleft, xright, kind='linear', **kwargs):
  from scipy.interpolate import interp1d
  sel = (xleft<=ux) & (ux<=xright)
  nx = len(ux[sel])
  fy = interp1d(ux[sel], uym[sel], kind=kind)
  finex = np.linspace(xleft, xright, 10*nx)
  finey = fy(finex)
  kwargs['marker'] = ''
  line = ax.plot(finex, finey, **kwargs)
  return line

def show_slice1d(ax, ux, uym, uye, kfl, kinds=None, efunc=None, **kwargs):
  if ('c' not in kwargs) and ('color' not in kwargs):
    raise RuntimeError('color must be specified')
  lines = []
  # set some default errorbar styles
  eb_kwargs = kwargs.copy()
  eb_kwargs['ls'] = ''
  # set default interpolation style to linear
  if kinds is None:
    kinds = ['linear'] * (len(kfl)+1)
  if efunc is None:
    def efunc(ax, *args, **kwargs):
      return ax.errorbar(*args, **kwargs)
  # spline and connect pieces of the 1D curve
  kleft = min(ux)
  for kf, kind in zip(kfl+[max(ux)], kinds):
    ksel = ux<=kf
    kright = max(ux[ksel])
    line0 = show_data(ax, efunc, ux, uym, uye, kleft, kright, **eb_kwargs)
    lines.append(line0)
    #line = show_spline(ax, ux, uym, kleft, kright, **kwargs)
    line = show_interp(ax, ux, uym, kleft, kright, kind=kind, **kwargs)
    lines.append(line)
    xnext = ux[~ksel]
    if len(xnext) > 0:
      kleft = min(xnext)
  return lines
