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
