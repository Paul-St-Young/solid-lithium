import numpy as np

def continue_clamped(x, y, kind='cubic'):
  """ continue a clamped spline to +/- infinity

  Args:
    x (np.array): x values
    y (np.array): y values
    kind (str, optional): spline kind, default 'cubic'
  Return:
    function: ycont, y(x) defined over all x
  """
  from scipy.interpolate import interp1d
  fy = interp1d(x, y, kind=kind)

  def fycont(p):
    # set function value to zero beyond interpolation range
    ycont = np.zeros(len(p))
    # use interpolating function inside interpolation range
    sel = (p > min(x)) & (p < max(x))
    ycont[sel] = fy(p[sel])
    return ycont
  return fycont

def flip_and_glue(x, y, xmult=-1, ymult=1):
  """ flip and glue a 1D samples
  the default works for symmetric function sampled on
  positive domain

  Args:
    x (np.array): x values
    y (np.array): y values
    xmult (float, optional): default -1 for positive domain
    ymult (float, optional): default 1 for symmetric function
  Return:
    (np.array, np.array): (myx, myy) glued samples
  """
  istart = 0
  if np.isclose(x[0], 0):
    istart = 1
  myk = np.concatenate([x*xmult, x[istart:]])
  myjp = np.concatenate([y*ymult, y[istart:]])
  idx = np.argsort(myk)
  return myk[idx], myjp[idx]

def flip_and_clamp(x, y, kind='cubic'):
  """ flip and glue a 1D function on the positive domain
  then cubic spline with clamped boundary conditions

  Args:
    x (np.array): x values
    y (np.array): y values
  Return:
    function: y(x) defined over (-max(x), max(x))
  """
  myx, myy = flip_and_glue(x, y)
  fy = continue_clamped(myx, myy, kind=kind)
  return fy

def lorentz(x, gamma):
  """ Lorentzian function """
  deno = x**2+(gamma/2.)**2
  mult = 1./np.pi*gamma/2
  return mult/deno

def elorentz(x, gamma, a1, a2):
  """ extended Lorentzian function """
  arg = 2*x/gamma
  deno = 1.+a1*arg**2+a2*arg**4
  # normalize
  delta = a1**2-4*a2
  assert delta > 0
  norm = np.pi/2*gamma/np.sqrt(2*delta)*(
    np.sqrt(a1+np.sqrt(delta)) - np.sqrt(a1-np.sqrt(delta))
  )
  return 1./deno/norm

def fft_convolve(f, g, pp):
  """ convolve two functions f and g on a discrete grid pp using FFT

  Args:
    f (function): float->float
    g (function): float->float
    pp (np.array): x values
  Return:
    np.array: convoluved function f*g sampled on grid pp
  """
  ft_fg = np.fft.fft(f(pp))*np.fft.fft(g(pp))
  norm = (pp.max()-pp.min())/len(pp)
  return np.fft.fftshift(np.fft.ifft(ft_fg).real)*norm

def qexp(x, q):
  if np.isclose(q, 1):
    return np.exp(x)
  return (1.+(1.-q)*x)**(1./(1-q))

def qgaussian_nonorm(x, q, beta):
  return qexp(-x**2/(2*beta**2), q)

def clement_hf_mo():
  def chi0(r, z, n):
    """S basis function"""
    from scipy.misc import factorial
    y00 = 0.5/np.sqrt(np.pi)  # Ylm l=0, m=0
    norm = y00/np.sqrt(factorial(2*n))* (2*z)**(n+.5)
    return norm*r**(n-1)*np.exp(-z*r)
  def mo(r):
    z1 = 2.47673
    z2 = 4.69873
    c1 = 0.89786
    c2 = 0.11131
    return c1*chi0(r, z1, 1) + c2*chi0(r, z2, 1)
  return mo

def clement_hf_nk(myk, rmin=0, rmax=20, nr=1024*16):
  from qharv.inspect.grsk import ft_iso3d
  finer = np.linspace(rmin, rmax, nr)
  mo = clement_hf_mo()
  psik = ft_iso3d(myk, finer, mo(finer))
  norm0 = 245.66/2  # converged norm using default rgrid
  return psik**2/norm0

def cubic_average(jp100, jp110, jp111):
  return 1./35*(10*jp100+16*jp110+9*jp111)

def scaled_jp1d(uk, unk, rs, rs1):
  from solith.li_nofk.sum_rule import kvol3d
  from solith.li_nofk.fit_nofk import heg_kfermi
  from solith.li_nofk.int_nofk import calc_jp1d
  #rs1 = heg_kfermi(kf1)
  kvol1 = kvol3d(rs1)
  # scale k
  kf = heg_kfermi(rs)
  kf1 = heg_kfermi(rs1)
  uk1 = uk*kf1/kf
  # scale J(p)
  unk1 = unk.copy()
  ujp1 = calc_jp1d(uk1, unk1)/kvol1
  return uk1, ujp1
