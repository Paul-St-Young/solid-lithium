import os
import numpy as np
from qharv.seed import wf_h5
from qharv.inspect import axes_pos

def get_momenta_and_weights(fp, ispin, ikpt, kmax, ecore, efermi):
  """ Histogram psig^2 for a single determinant of Kohn-Sham orbitals at a
  single twist. Select states with momenta k<kmax and energy ecore<e<efermi.

  Args:
    fp (h5py.File): pwscf.pwscf.h5 file pointer
    ispin (int): determinant index (0: unpolarized, 0/1: polarized)
    ikpt (int): twist vector index
    kmax (float): maximum momentum to record
    ecore (float): Core state energy bound. States below ecore are excluded
    efermi (float): Fermi energy
  Return:
    (np.array, np.array): (momenta, weights)
  """
  # get reciprocal lattice vectors to do lattice transformation
  axes = wf_h5.get(fp, 'axes')
  raxes = axes_pos.raxes(axes)
  # Bloch function momenta (kvecs)
  gvecs = wf_h5.get(fp, 'gvectors')
  kvecs = np.dot(gvecs, raxes)
  # crystal momentum (tvec)
  kpath = wf_h5.kpoint_path(ikpt)
  utvec = fp[os.path.join(kpath, 'reduced_k')].value
  tvec = np.dot(utvec, raxes)
  # true momentum = crystal + Bloch momentum
  mykvecs = kvecs+tvec[np.newaxis, :]

  # keep kvectors below kmax
  mykmags = np.linalg.norm(mykvecs, axis=1)
  sel = mykmags < kmax
  momenta = mykvecs[sel]

  # accumulate occupation
  weights = np.zeros(len(momenta))
  # keep states below the Fermi level and outside the core
  nstate = fp[os.path.join(kpath, 'spin_%d'%ispin, 'number_of_states')].value[0]
  evals = fp[os.path.join(kpath, 'spin_%d'%ispin, 'eigenvalues')].value
  esel = (evals>ecore) & (evals<efermi)
  states = np.arange(nstate)
  for istate in states[esel]:
    psig = wf_h5.get_orb_in_pw(fp, ikpt, ispin, istate)
    pg2 =  psig.conj()*psig
    if not np.allclose(pg2.imag, 0): raise RuntimeError('dot not zero')
    pg2 = pg2.real[sel]
    weights += pg2
  return momenta, weights


def get_momentum_distribution(fp, kmax, efermi, ispin=0, ecore=-np.inf):
  """ obtain momentum distribution from Kohn-Sham determinant stored 
  in QMCPACK wf.h5 file format

  Args:
    fp (h5py.File): pwscf.pwscf.h5 file pointer
    kmax (float): maximum k magnitude to record
    efermi (float): Fermi energy
  Return:
    (np.array, np.array): (kvecs, nkm), kvectors and n(k) mean (no error)
  """
  nkpt = wf_h5.get(fp, 'nkpt')[0]
  mlist = []
  wlist = []
  for ikpt in range(nkpt):
    momenta, weights = get_momenta_and_weights(fp, ispin, ikpt, kmax,
      ecore, efermi)
    mlist += momenta.tolist()
    wlist += weights.tolist()
  # end for ikpt
  kvecs = np.array(mlist)
  nkm = np.array(wlist)
  return kvecs, nkm

def get_det_nk(fp, efermi, ecore=-np.inf, kmax=np.inf, ispin=0):
  """Calculate momentum distribution from Kohn-Sham determinant

  same as get_momentum_distribution, but faster

  Args:
    fp (h5py.File): pwscf.pwscf.h5 file pointer
    efermi (float): Fermi energy
    ecore (float, optional): core energy, default -np.inf
    kmax (float, optional): maximum k magnitude to record, default np.inf
    ispin (int, optiona): default 0
  Return:
    (np.array, np.array): (kvecs, nkm), kvectors and n(k) mean (no error)
  """
  from solith.li_nofk.forlib.det import nofk
  # need Kohn-Sham eigenvalues to decide which states to use
  bands = wf_h5.get_bands(fp, ispin=ispin)
  nt, nstate = bands.shape

  # need kgrid info: basis (raxes), unshifted (kvecs0), twists (tvecs)
  axes = wf_h5.get(fp, 'axes')
  raxes = axes_pos.raxes(axes)
  gvecs = wf_h5.get(fp, 'gvectors')
  utvecs = wf_h5.get_twists(fp)
  tvecs = np.dot(utvecs, raxes)
  kvecs0 = np.dot(gvecs, raxes)

  # calculate n(k) at each twist
  kvecsl = []
  nkml = []
  for it in range(nt):
    kvecs = kvecs0 + tvecs[it]  # current twist
    # histogram orb^2
    cmat = wf_h5.get_cmat(fp, it, ispin, nstate)
    nocc, npw = cmat.shape
    weights = nofk(kvecs, cmat, bands[it], kmax, ecore, efermi)
    # save within a cutoff
    kmags = np.linalg.norm(kvecs, axis=-1)
    sel = kmags < kmax
    kvecsl.append(kvecs[sel])
    nkml.append(weights[sel])

  kvecs = np.concatenate(kvecsl, axis=0)
  nkm = np.concatenate(nkml)
  return kvecs, nkm
