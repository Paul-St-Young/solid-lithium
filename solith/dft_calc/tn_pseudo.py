import numpy as np


def read_tags_and_datablocks(text):
  """ read a file consisting of blocks of numbers which are
  separated by tag lines. separate tag lines from data lines
  return two lists

  e.g. for pp.data file:
Atomic number and pseudo-charge
14 4.00
Energy units (rydberg/hartree/ev):
rydberg
Angular momentum of local component (0=s,1=p,2=d..)
2
NLRULE override (1) VMC/DMC (2) config gen (0 ==> input/default value)
0 0
Number of grid points
1603
R(i) in atomic units
    0.000000000000000E+00
    0.719068853804059E-09
    0.144778949458300E-08

  will be parsed into:
   ['Atomic number and pseudo-charge', ...
   ['14 4.00', ...

  Args:
    text (str): file content
  Return:
    tuple: (tags, blocks), both are a list of strings
  """
  lines = text.split('\n')
  tags = []
  blocks = []

  block = ''
  for line in lines:
    try:
      map(float, line.split())
      block += line
    except:
      tags.append(line)
      blocks.append(block)
      block = ''
  blocks.append(block)
  return tags, blocks[1:]


def parse_pp_data(pp_data):
  """ parse Trail-Needs pp.data file for radial grid and potential

  Args:
    pp_data (str): pp.data filename
  Return:
    tuple: (grid, potl) grid is the radial grid, potl is a list of
     r*Vloc
  """
  with open(pp_data, 'r') as f:
    entryl, blockl = read_tags_and_datablocks(f.read())
  potl = []
  for label, block in zip(entryl, blockl):
    if label.startswith('R(i)'):
      grid = map(float, block.split())
    if label.startswith('r*potential'):
      pot = map(float, block.split())
      potl.append(pot)
  return grid, potl


def parse_awfn_data(awfn_data):
  """ parse Trail-Needs awfn.data* file for radial grid and orbital

  Args:
    awfn_data (str): awfn.data* fname
  Return:
    tuple: (grid, orbl) grid is the radial grid, orbl is a list of
     orbitals
  """
  with open(awfn_data, 'r') as f:
    entryl, blockl = read_tags_and_datablocks(f.read())
  orbl = []
  for label, line in zip(entryl, blockl):
    if label.startswith('Radial grid'):
      rgrid = map(float, line.split()[1:])
    if label.startswith('Orbital #'):
      orb = map(float, line.split()[3:])
      orbl.append(orb)
  return rgrid, orbl


def parse_upf(pp_upf):
  """ parse UPF pseudopotential file
  return parse data in a dictionary

  Args:
    pp_upf (str): upf filename
  Return:
    dict: a dictionary of parsed data
  """
  from qharv.seed import xml
  # parse like xml file
  with open(pp_upf, 'r') as f:
    body = f.read()
  text_xml = '<root>' + body + '</root>'
  doc = xml.parse(text_xml)

  # read mesh
  node = doc.find('.//PP_R')
  line = ' '.join(node.text.split('\n'))
  ugrid = np.array(line.split(), dtype=float)

  # read local
  node = doc.find('.//PP_LOCAL')
  line = ' '.join(node.text.split('\n'))
  uloc = np.array(line.split(), dtype=float)

  # read non-local
  nodes = doc.findall('.//PP_BETA')
  unlocl = []
  for node in nodes:
    line = ' '.join(node.text.split('\n')[3:])
    unloc = np.array(line.split(), dtype=float)
    unlocl.append(unloc.tolist())

  node = doc.find('.//PP_PSWFC')
  tags, data = read_tags_and_datablocks(node.text)
  mat = np.array([map(float, line.split()) for line in data], dtype=float)

  node = doc.find('.//PP_RHOATOM')
  mat1 = np.array(' '.join(node.text.split('\n')).split(), dtype=float)

  info_dict = {
    'grid': ugrid.tolist(),
    'vloc': uloc.tolist(),
    'vnl_list': unlocl,
    'pp_pswfc_meta': tags,
    'pp_pswfc': mat.tolist(),
    'rhoatom': mat1.tolist()
  }
  return info_dict


def parse_pp_dat(pp_dat):
  data = np.loadtxt(pp_dat)
  grid = data[:, 0]
  vmat = data[:, 1:]
  return grid, vmat.T
