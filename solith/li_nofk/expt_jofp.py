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
