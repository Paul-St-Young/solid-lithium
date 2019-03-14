import numpy as np

# Paola PRB 66, 235116 (2002)
def paola_nm(rs):
  v1 = -0.0679793
  v2 = -0.00102846
  v3 = 0.000189111
  v4 = 0.0205397
  v5 = -0.0086838
  v6 = 6.87109e-5
  v7 = 4.868047e-5
  nume = 1+v1*rs+v2*rs**2+v3*rs**3
  deno = 1+v4*rs+v5*rs**2+v6*rs**3+v7*rs**(15./4)
  return nume/deno
def paola_np(rs):
  q1 = 0.088519
  q2 = 0.45
  q3 = 0.022786335
  nume = q1*rs
  deno = 1+q2*rs**0.5+q3*rs**(7./4)
  return nume/deno
def paola_zkf(rs):
  nminus = paola_nm(rs)
  nplus = paola_np(rs)
  return nminus-nplus

# finite-size correction for Zkf
def rpa_delta3d(k, rs):
  wp = (3./rs**3)**0.5
  alpha = 4*np.pi/wp
  beta = alpha/wp
  return 0.5*(alpha/k**2-beta)
def get_qvecs_and_intnorm(axes, mx):
  from qharv.inspect import axes_pos
  # get qvectors
  raxes = axes_pos.raxes(axes)
  fvecs = 1./mx*axes_pos.cubic_pos(mx)
  qvecs = np.dot(fvecs, raxes)
  qvecs -= qvecs.mean(axis=0)
  # calculate integration norm
  intnorm = axes_pos.volume(raxes)/mx**3/(2*np.pi)**3
  return qvecs, intnorm
def integrate_in_missing_volume(axes, mx, func3d):
  qvecs, intnorm = get_qvecs_and_intnorm(axes, mx)
  intval = func3d(qvecs).sum()
  return intval*intnorm
def get_rpa_delta3d_inf(axes, rs, mxl=[8, 16], alpha=-1.):
  from qharv.sieve.scalar_df import poly_extrap_to_x0

  def rpa_d3d(qvecs):  # turn 1D isotropic function to 3D function
    qmags = np.linalg.norm(qvecs, axis=-1)
    return rpa_delta3d(qmags, rs)

  # extrapolate to infinite quadrature grid
  yl = []
  for mx in mxl:
    d3d = integrate_in_missing_volume(axes, mx, rpa_d3d)
    yl.append(d3d)
  ye = [1e-8]*len(yl)  # fake errors
  y0m, y0e = poly_extrap_to_x0(np.array(mxl)**alpha, yl, ye, 1)
  return y0m, y0e
# 1-term approximation to rpa_delta3d
def delta3d_1term(rs, nelec):
  return 1.21843*(rs/3.)**0.5*(3./(4*np.pi*nelec))**(1./3)
