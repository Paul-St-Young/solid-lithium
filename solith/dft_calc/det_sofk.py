import os
import numpy as np

def write_detsk(h5file, ikpt, fwf, ispin, nsh0, kc):
  """Calculate determinant S(k) at given twist

  Args:
    h5file (tables.file.File): hdf5 file handle
    ikpt (int): twist index
    fwf (str): wf h5 file e.g. pwscf.pwscf.h5
    ispin (int): spin index, use 0 for unpolarized
    nsh0 (int): number of shells to use
    kc (float): PW cutoff
  """
  from qharv.seed import wf_h5
  from qharv.inspect.axes_pos import raxes
  from qharv.reel.config_h5 import save_dict
  from solith.li_nofk.forlib.det import calc_detsk
  from chiesa_correction import mirror_xyz, cubic_pos
  # creat slab for twist
  gname = 'twist%s' % str(ikpt).zfill(3)
  slab = h5file.create_group('/', gname, '')

  # read wf file
  fp = wf_h5.read(fwf)
  gvecs, cmat = wf_h5.get_cmat(fp, ikpt, ispin)
  wf_h5.normalize_cmat(cmat)
  axes = wf_h5.get(fp, 'axes')
  fp.close()
  raxes = raxes(axes)

  # decide which qvectors to calculate S(q)
  qvecs = mirror_xyz(cubic_pos(nsh0))
  kvecs = np.dot(qvecs, raxes)
  kmags = np.linalg.norm(kvecs, axis=-1)
  qsel = (1e-8 < kmags) & (kmags < kc)

  # calculate S(k)
  sk0 = calc_detsk(qvecs[qsel],  gvecs, cmat)

  # save
  arr_dict = {
    'raxes': raxes,
    'gvecs': qvecs[qsel],
    'sk0': sk0
  }
  save_dict(arr_dict, h5file, slab)

def get_kssk(fh5, remove_bragg=False):
  """Retrieve determinant S(k) from HDF5 archive

  Args:
    fh5 (str): hdf5 archive
    remove_bragg (bool, optional): remove Bragg peaks
  Return:
    (np.array, np.array, np.array): raxes, gvecs, sk0a
      reciprocal lattice, integer vectors, twist-resolved S(k)
  """
  import h5py
  fp = h5py.File(fh5)
  # get kvectors at one twist
  twist = fp.keys()[0]
  gpath = os.path.join(twist, 'gvecs')
  rpath = os.path.join(twist, 'raxes')
  raxes = fp[rpath].value
  gvecs = fp[gpath].value
  # average sk overall twists
  skl = []
  sel = np.ones(len(gvecs), dtype=bool)
  for twist in fp.keys():
    mpath = os.path.join(twist, 'sk0')
    sk0 = fp[mpath].value
    if remove_bragg:
      sel = sel & (sk0 <= 1.)
    skl.append(sk0)
  fp.close()
  ska = np.array(skl)[:, sel]
  return raxes, gvecs[sel], ska
