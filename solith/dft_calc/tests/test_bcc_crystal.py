import numpy as np
from solith.dft_calc import bcc_crystal as bc

def test_n54():
  axes = bc.get_cubic_axes(3.25, 54)
  alat = axes[0][0]
  assert np.isclose(alat, 19.8020803)

def test_n432():
  axes = bc.get_cubic_axes(3.25, 432)
  alat = axes[0][0]
  assert np.isclose(alat, 19.8020803*2)
