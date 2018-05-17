# read .cfg file. Extend by:
#  1. add an item to parse_funs
#  2. append a parse function to this script
import numpy as np

from qharv.seed.xml import text2arr
from qharv.reel import ascii_out, mole


def read_cfg(fcfg):
  mm = ascii_out.read(fcfg)

  idxl = ascii_out.all_lines_with_tag(mm, 'ITEM:')
  data = {}
  for idx in idxl:
    mm.seek(idx)
    header = mm.readline()
    item_name = header.split(':')[-1].strip()

    entry = parse_item(mm, item_name)
    data.update(entry)
  return data


def parse_item(mm, item_name):
  parse_funcs = {
    'TIMESTEP':parse_timestep,
    'NUMBER OF ATOMS':parse_natom,
    'ATOMS id x y z':parse_ixyz,
    'BOX BOUNDS pp pp pp':parse_box_ppp
  }
  known_items = parse_funcs.keys()
  entry = {}
  if item_name in known_items:
    entry = parse_funcs[item_name](mm)
  return entry


def parse_timestep(mm):
  line = mm.readline()
  nstep = int(line)
  return {'nstep':nstep}


def parse_natom(mm):
  line = mm.readline()
  natom = int(line)
  return {'natom':natom}


def parse_ixyz(mm):
  text = mm[mm.tell():]
  mat = text2arr(text)
  pos = mat[mat[:, 0].argsort(), 1:]
  return {'pos':pos.tolist()}


def parse_box_ppp(mm):
  ndim = 3
  alatl = []
  for idim in range(ndim):
    line = mm.readline()
    low, high = map(float, line.split())
    lbox = high - low
    alatl.append(lbox)
  # end for
  axes = np.diag(alatl)
  return {'axes':axes}
