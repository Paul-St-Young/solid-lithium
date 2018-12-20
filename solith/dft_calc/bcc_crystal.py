import numpy as np

def rs2alat(rs):
  return (8.*np.pi/3.)**(1./3)*rs

def alat2rs(alat):
  (3./(8.*np.pi))**(1./3)*alat

def get_cubic_axes(rs, natom):
  alat = rs2alat(rs)
  volfac = natom/2.  # volume multiplier from conventional cell
  nx = volfac**(1./3)  # (nx, nx, nx) supercell of conv. cell
  if not np.isclose(round(nx), nx):
    raise RuntimeError('cannot handle natom = %d; nx=%3.2f' % (natom, nx))
  axes = nx*alat*np.eye(3)
  return axes
