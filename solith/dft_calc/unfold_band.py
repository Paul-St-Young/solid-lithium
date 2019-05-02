import numpy as np

def get_bands(nscf_out, tgrid0):
  """Get bands and kgrid info from nscf output

  data contains: kvecs, bands, tgrid, raxes, gvecs
  kvecs (nk, ndim) are reciprocal points possible in the irreducible wedge
   kvecs are in 2\pi/alat units
  bands (nk, nstate) are the Kohn-Sham eigenvalues
   bands are in eV units
  tgrid (ndim) is grid size in each dimension
   !!!! currently assumed to be the same as x
  raxes (ndim, ndim) is the reciprocal lattice
  gvecs (nk, ndim) are reciprocal lattice points (kvecs) converted to integers

  Args:
    nscf_out (str): output file
    tgrid0 (int): grid along x
  Return:
    dict: data
  """
  from qharv.inspect import axes_pos
  import qe_reader as qer
  # get bands
  data = qer.parse_nscf_bands(nscf_out)
  kvecs = data['kvecs']

  # get raxes, gvecs
  tgrid = np.array([tgrid0]*3)
  axes = qer.read_out_cell(nscf_out)
  raxes = axes_pos.raxes(axes)

  gcand = np.dot(kvecs, np.linalg.inv(raxes/tgrid))
  gvecs = np.around(gcand).astype(int)

  data['tgrid'] = tgrid
  data['raxes'] = raxes
  data['gvecs'] = gvecs
  data.pop('nkpt')
  return data

def get_ekmap(scf_out):
  """Obtain the internal variable 'equiv' from kpoint_grid.f90 in QE/PW

  store the maps between full BZ (fBZ) and irreducible BZ (iBZ)

  Args:
    scf_out (str): output file
  Return:
    (dict, dict): (fBZ->iBZ, iBZ->fBZ) maps
  """
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)
  text = ascii_out.block_text(mm, 'equivalent kpoints begin', 'end')
  lines = text.split('\n')
  emap = {}  # full kgrid to irreducible wedge
  kmap = {}  # irreducible wedge to full kgrid
  for line in lines:
    tokens = line.split('equiv')
    if len(tokens) != 2: continue
    left, right = map(int, tokens)
    emap[left] = right
    if right in kmap:
      kmap[right].append(left)
    else:
      kmap[right] = [left]
  mm.close()
  return emap, kmap

def get_weights(equiv_out):
  """Get weights of irreducible kpoints.

  Args:
    equiv_out (str): QE output file
  Return:
    np.array: weights, number of equivalent kpoints for each irrek
  """
  emap, kmap = get_ekmap(equiv_out)
  sidxl = kmap.keys()
  sidxl.sort()
  weights = []
  for sidx in sidxl:
    kwt = len(kmap[sidx])
    weights.append(kwt)
  return np.array(weights)

def unfold2(bands, emap, kmap, axis=0):
  """unfold method 2: steal equivalence map from QE kpoint_grid.f90

  kpoints in bands MUST be ordered in the same way as the QE irreducible kpts

  Args:
    bands (np.array): band energy with kpoint (and state) labels
    emap (dict): int -> int equivalence map of kpoint indices (full -> irrek)
    kmap (dict): inverse of emap
    axis (int, optional): kpoint axis, default is 0
  Return:
    np.array: unfolded bands
  """
  idxl = kmap.keys()
  idxl.sort()
  nktot = len(emap)
  # extend the kpoint axis
  new_shape = list(bands.shape)
  new_shape[axis] = nktot
  vals = np.zeros(new_shape)
  # fill existing values
  for i, idx in enumerate(idxl):
    if axis == 0:
      vals[idx-1] = bands[i]
    elif axis == 1:
      vals[:, idx-1] = bands[:, i]
    else:
      raise RuntimeError('need to implement axis %d (add another :,)' % axis)
  # map symmetry points
  for idx0, idx1 in emap.items():
    if axis == 0:
      vals[idx0-1] = vals[idx1-1]
    elif axis == 1:
      vals[:, idx0-1] = vals[:, idx1-1]
  return vals

def get_mats_vecs(symops):
  mats = []
  vecs = []
  for so in symops:
    mat = np.array(so['mat'], int)
    vec = np.array(so['vec'], int)
    mats.append(mat)
    vecs.append(vec)
  return np.array(mats), np.array(vecs)

def unfold1(gvecs1, nkm1, nscf_out, pbc, show_progress=True):
  """unfold method 1: apply symmetry operations to unique gvecs

  notice, there is no reason to carry nkm1 around
  todo: unfold kgrid only, one symmetry operation at a time
   return a list of 1D indices on the regular grid

  Args:
    gvecs1 (np.array): integer vectors in the irreducible BZ
    nkm1 (np.array): scalar field defined over gvecs1
    scf_out (str): nscf output containing symmetry matrices
    pbc (bool): apply periodic boundary condition
    show_progress (bool, optional): show progress bar, default True
  """
  # get symops
  import qe_reader as qer
  symops = qer.read_sym_ops(nscf_out)

  # make a grid large enough to contain the unfolded n(k)
  import chiesa_correction as chc
  gmin, gmax, ng = chc.get_regular_grid_dimensions(gvecs1)
  rgvecs = chc.get_regular_grid(gmin, gmax, ng, int)

  # unfold
  rnkm = np.zeros(len(rgvecs))
  filled = np.zeros(len(rgvecs), dtype=bool)
  if show_progress:
    from progressbar import ProgressBar
    bar = ProgressBar(maxval=len(symops))
  for isym, so in enumerate(symops):
    mat = np.array(so['mat'], dtype=int)
    for ig, gv in enumerate(gvecs1):  # unfold existing data
      gv1 = np.dot(mat, gv)
      if pbc:
        # bring back gvectors outside of rgvecs
        gv1 = (gv1-gmin) % ng + gmin
      else:
        # ignore gvectors outside of rgvecs
        if (gv1 < gmin).any() or (gv1 > gmax).any(): continue
      idx3d = gv1-gmin
      # save new point
      idx = np.ravel_multi_index(idx3d, ng)
      if not filled[idx]:
        rnkm[idx] = nkm1[ig]
        filled[idx] = True
    bar.update(isym)
  return rgvecs[filled], rnkm[filled]

def unfold_idx(gvecs1, mats, pbc):
  # make a grid large enough to contain the unfolded n(k)
  import chiesa_correction as chc
  gmin, gmax, ng = chc.get_regular_grid_dimensions(gvecs1)
  rgvecs = chc.get_regular_grid(gmin, gmax, ng, int)

  # unfold
  npt = np.prod(ng)
  filled = np.zeros(npt, dtype=bool)
  ridx = np.ones(npt, dtype=int)
  for mat in mats:
    for ig, gv in enumerate(gvecs1):  # unfold existing data
      gv1 = np.dot(mat, gv)
      if pbc:
        # bring back gvectors outside of rgvecs
        gv1 = (gv1-gmin) % ng + gmin
      else:
        # ignore gvectors outside of rgvecs
        if (gv1 < gmin).any() or (gv1 > gmax).any(): continue
      idx3d = gv1-gmin
      # save new point
      idx = np.ravel_multi_index(idx3d, ng)
      if not filled[idx]:
        filled[idx] = True
        ridx[idx] = ig
  ridx[~filled] = -1
  return rgvecs, ridx

def compare_scalar_grids(gvecs0, nkm0, gvecs1, nkm1, atol=1e-6):
  """Compare two scalar fields sampled on regular grids

  Args:
    gvecs0 (np.array): first grid, (npt0, ndim)
    nkm0 (np.array): values, (npt0,)
    gvecs1 (np.array): second grid, (npt1, ndim), expect npt1<=npt0
    nkm1 (np.array): values, (npt1,)
  Return:
    bool: True if same scalar field
  """
  from chiesa_correction import align_gvectors
  comm0, comm1 = align_gvectors(gvecs0, gvecs1)
  unique = len(gvecs1[comm1]) == len(gvecs1)  # all unique gvecs are unique
  xmatch = np.allclose(gvecs0[comm0], gvecs1[comm1],
    atol=atol)  # gvecs match
  ymatch = np.allclose(nkm0[comm0], nkm1[comm1],
    atol=atol)  # nk match before unfold
  return np.array([unique, xmatch, ymatch], dtype=bool)
