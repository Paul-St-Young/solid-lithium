import os
import numpy as np
from qharv.seed import wf_h5
from qharv.inspect import axes_pos

def get_momenta_and_weights(fp, ikpt, kmax, efermi, ispin=0):
  """ record all momenta and psig^2 for k<kmax and e<efermi

  Args:
    fp (h5py.File): pwscf.pwscf.h5 file pointer
    ikpt (int): twist vector index
    kmax (float): maximum momentum to record
    efermi (float): Fermi energy
    ispin (int, optional): determinant index, default 0
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
  # keep states below the Fermi level
  nstate = fp[os.path.join(kpath, 'spin_%d'%ispin, 'number_of_states')].value[0]
  evals = fp[os.path.join(kpath, 'spin_%d'%ispin, 'eigenvalues')].value
  esel = evals < efermi
  states = np.arange(nstate)
  for istate in states[esel]:
    psig = wf_h5.get_orb_in_pw(fp, ikpt, ispin, istate)
    pg2 =  psig.conj()*psig
    if not np.allclose(pg2.imag, 0): raise RuntimeError('dot not zero')
    pg2 = pg2.real[sel]
    weights += pg2
  return momenta, weights


def get_momentum_distribution(fp, kmax, efermi):
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
    momenta, weights = get_momenta_and_weights(fp, ikpt, kmax, efermi)
    mlist += momenta.tolist()
    wlist += weights.tolist()
  # end for ikpt
  kvecs = np.array(mlist)
  nkm = np.array(wlist)
  return kvecs, nkm