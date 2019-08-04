import numpy as np

def kvol3d(rs):
  """Volume of reciprocal state per electron in 3D (unpolarized)

  Args:
    rs (float): Wigner-Seitz density parameter
  Return:
    float: kvol
  """
  kvol = (2*np.pi)**3/(4*np.pi*rs**3/3)
  return kvol

def ntsum_raw3d(kvecs, nkm, kvol, nke=None):
  """ Calculate momentum distribution sum rules using raw n(k) in 3D

  !!!! kvecs must be on a simple cubic lattice

  Args:
    kvecs (np.array): kvectors
    nkm (np.array): momentum distribution mean
    kvol (float): reciprocal space volume of a state, norm=1/kvol
    nke (np.array, optional): momentum distribution error, default None
  Return:
    (float, float): (nsum, tsum)
  """
  kx = np.unique(kvecs[:, 0])
  kx.sort()
  dk = kx[1]-kx[0]
  kmags = np.linalg.norm(kvecs, axis=-1)
  nsum = dk**3*nkm.sum()/kvol
  tsum = dk**3*(0.5*kmags**2*nkm).sum()/kvol
  if nke is not None:
    nsume = dk**3*(nke**2).sum()**0.5/kvol
    tsume = dk**3*(0.5*kmags**2*nke**2).sum()**0.5/kvol
    return nsum, nsume, tsum, tsume
  return nsum, tsum

def ntsum_iso3d(uk, unkm, kvol, nke=None):
  """ Calculate momentum distribution sum rules using spherically-averaged
  isotropic n(k) in 3D

  Args:
    uk (np.array): unique kvector magnitudes
    unkm (np.array): momentum distribution mean
    kvol (float): reciprocal space volume of a state, norm=1/kvol
    nke (np.array, optional): momentum distribution error, default None
  Return:
    (float, float): (nsum, tsum)
  """
  nsum = 4*np.pi*np.trapz(uk**2*unkm, uk)/kvol
  tsum = 4*np.pi*np.trapz(0.5*uk**4*unkm, uk)/kvol
  if nke is not None:
    nsume = 4*np.pi*np.trapz(uk**2*nke**2, uk)**0.5/kvol
    tsume = 4*np.pi*np.trapz(0.5*uk**4*nke**2, uk)**0.5/kvol
    return nsum, nsume, tsum, tsume
  return nsum, tsum

def nktail(k, A, Z):
  nume = (k**2+Z**2)**2
  return A*(2*Z/nume)**2

def jptail(p, A, Z, kglue, kmax=10., nk=1024):
  from solith.li_nofk.int_nofk import calc_jp1d
  from solith.li_nofk.expt_jofp import flip_and_clamp
  finek = np.linspace(kglue, kmax, nk)
  finenk = nktail(finek, A, Z)
  yz = np.zeros(len(p))
  finex = np.concatenate([p, finek], axis=0)
  finey = np.concatenate([yz, finenk], axis=0)
  dy = calc_jp1d(finex, finey)
  fdjp = flip_and_clamp(finex, dy, kind='linear')
  return fdjp(p)
