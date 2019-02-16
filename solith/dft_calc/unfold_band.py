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

def grid_emap(emap):
  y = emap.values()
  ymap = {}
  for i, yy in enumerate(np.unique(y)):
    ymap[yy] = i
  sites = np.zeros(len(y), dtype=int)
  for xx, yy in emap.items():
    sites[xx-1] = ymap[yy]
  return np.array(sites)

def unfold2(gvecs1, nkm1, kmap_out, tgrid0):
  import chiesa_correction as chc
  emap, kmap = get_ekmap(kmap_out)
  sites = grid_emap(emap)
  nkm0 = nkm1[sites]
  qe_gvecs = chc.cubic_pos(tgrid0)/float(tgrid0)
  ukvecs0 = (qe_gvecs+.5) % 1 - .5
  gvecs0 = np.around(ukvecs0*tgrid0).astype(int)
  return gvecs0, nkm0

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
