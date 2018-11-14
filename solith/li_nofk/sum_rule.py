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

def ntsum_raw3d(kvecs, nkm, kvol):
  """ Calculate momentum distribution sum rules using raw n(k) in 3D

  !!!! kvecs must be on a simple cubic lattice

  Args:
    kvecs (np.array): kvectors
    nkm (np.array): momentum distribution mean
    kvol (float): reciprocal space volume of a state, norm=1/kvol
  Return:
    (float, float): (nsum, tsum)
  """
  kx = np.unique(kvecs[:, 0])
  kx.sort()
  dk = kx[1]-kx[0]
  kmags = np.linalg.norm(kvecs, axis=-1)
  nsum = dk**3*nkm.sum()/kvol
  tsum = dk**3*(0.5*kmags**2*nkm).sum()/kvol
  return nsum, tsum

def ntsum_iso3d(uk, unkm, kvol):
  """ Calculate momentum distribution sum rules using spherically-averaged
  isotropic n(k) in 3D

  Args:
    uk (np.array): unique kvector magnitudes
    unkm (np.array): momentum distribution mean
    kvol (float): reciprocal space volume of a state, norm=1/kvol
  Return:
    (float, float): (nsum, tsum)
  """
  nsum = 4*np.pi*np.trapz(uk**2*unkm, uk)/kvol
  tsum = 4*np.pi*np.trapz(0.5*uk**4*unkm, uk)/kvol
  return nsum, tsum
