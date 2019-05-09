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

# Yasutami Takada PRB 44, 7879 (1991)
def takada_nm():
  # extract from Takada1991 Fig. 2 and 4
  #rs = np.array([1, 3, 4, 5])
  #nminus = np.array([0.930,       0.836, 0.783, 0.751])
  # extract from Paola Fig. 4
  rs = np.array([1, 2, 3, 4, 5])
  nminus = np.array([0.930, 0.868, 0.817, 0.772, 0.730])
  return rs, nminus
def takada_np():
  # extract from Takada1991 Fig. 2 and 4
  #rs = np.array([1, 3, 4, 5])
  #nplus = np.array([0.055,       0.127, 0.154, 0.179])
  # extract from Paola Fig. 4
  rs = np.array([1, 2, 3, 4, 5])
  nplus = np.array([0.056, 0.101, 0.138, 0.169, 0.197])
  return rs, nplus
def takada_zkf():
  rs, nminus = takada_nm()
  rs, nplus = takada_np()
  zkf = nminus-nplus
  #zkf = np.array([[0.875, 0.709, 0.629, 0.572])
  # use Paola Fig. 4
  rs = np.array([1, 2, 3, 4, 5])
  zkf = np.array([0.870, 0.765, 0.680, 0.604, 0.545])
  return rs, zkf

# Markus Holzmann PRL 107, 110402 (2011)
def holzmann_zkf(method='bf-rmc'):
  # Table II
  rs = np.array([1, 2, 3.99, 5, 10])
  if method == 'bf-rmc':
    zkfm = np.array([0.84, 0.77, 0.64, 0.58, 0.40])
    zkfe = np.array([0.02, 0.01, 0.01, 0.01, 0.01])
  elif method == 'bf-vmc':
    zkfm = np.array([0.86, 0.78, 0.65, 0.59, 0.41])
    zkfe = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
  elif method == 'sj-vmc':
    zkfm = np.array([0.894, 0.82, 0.69, 0.61, 0.45])
    zkfe = np.array([0.009, 0.01, 0.01, 0.02, 0.01])
  else:
    raise RuntimeError('unknown method %s' % method)
  return rs, zkfm, zkfe

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
