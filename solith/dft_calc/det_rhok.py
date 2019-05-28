import numpy as np

def get_rhok(fwf, kbrags, norb=None, itwist=0, ispin=0):
  from qharv.seed import wf_h5
  from qharv.inspect import axes_pos
  from rsgub.grids.forlib.find_gvecs import calc_rhok
  fp = wf_h5.read(fwf)
  axes = wf_h5.get(fp, 'axes')
  gvecs = wf_h5.get(fp, 'gvectors')
  cmat = wf_h5.get_cmat(fp, itwist, ispin, norb=norb)
  fp.close()
  # calculate ukbrags
  raxes = axes_pos.raxes(axes)
  kcand = np.dot(kbrags, np.linalg.inv(raxes))
  ukbrags = np.round(kcand).astype(int)
  # check ukbrags
  kbrags1 = np.dot(ukbrags, raxes)
  assert np.allclose(kbrags, kbrags1)
  # calculate rhok
  val = 2*calc_rhok(ukbrags, gvecs, cmat)
  return val

def get_all_rhok(fwf, kbrags, norb=None, ispin=0, show_progress=True):
  from qharv.seed import wf_h5
  from qharv.inspect import axes_pos
  from rsgub.grids.forlib.find_gvecs import calc_rhok
  fp = wf_h5.read(fwf)
  axes = wf_h5.get(fp, 'axes')
  # calculate ukbrags
  raxes = axes_pos.raxes(axes)
  kcand = np.dot(kbrags, np.linalg.inv(raxes))
  ukbrags = np.round(kcand).astype(int)
  # check ukbrags
  kbrags1 = np.dot(ukbrags, raxes)
  assert np.allclose(kbrags, kbrags1)
  # calculate rhok
  gvecs = wf_h5.get(fp, 'gvectors')
  ntwist = wf_h5.get(fp, 'nkpt')[ispin]
  # store complex numbers in real view
  data = np.zeros([ntwist, 2*len(ukbrags)])
  if show_progress:
    from progressbar import ProgressBar
    bar = ProgressBar(maxval=ntwist)
  for itwist in range(ntwist):
    cmat = wf_h5.get_cmat(fp, itwist, ispin, norb=norb)
    val = 2*calc_rhok(ukbrags, gvecs, cmat)
    data[itwist, :] = val.view(float)
    if show_progress:
      bar.update(itwist)
  fp.close()
  return data
