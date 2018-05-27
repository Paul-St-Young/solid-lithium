import numpy as np
from copy import deepcopy
from lxml import etree
from qharv.seed import xml, xml_examples


def pset_from_pos(pos):
  """ construct <particleset name="ion0"> from position array
  Args:
    pos (np.array): list of atomic positions
  Return:
    lxml.Element: <particleset>
  """
  name = 'Li'
  charge = 1  # [He] core BFD psp
  atomic_number = 3  # needed to check with pseudopotential

  natom = len(pos)

  pset = etree.Element('particleset', {'name': 'ion0'})
  group = etree.Element('group', {
    'name': 'Li',
    'size': str(natom)
  })
  xml.set_param(group, 'charge', ' ' + str(charge) + ' ', new=True)
  xml.set_param(group, 'atomicnumber', ' ' + str(atomic_number) + ' ', new=True)

  pos_arr = etree.Element('attrib', {
    'name': 'position',
    'datatype': 'posArray',
    'condition': '0'
  })
  pos_arr.text = xml.arr2text(pos)

  group.append(pos_arr)
  pset.append(group)
  return pset


def psp_ham(psp='Li.BFD.xml'):
  """ construct <hamiltonian> using given pseudopotential
  Args:
    pos (np.array): list of atomic positions
  Return:
    lxml.Element: <particleset>
  """
  name = 'Li'
  ham = xml_examples.static_coul_ae_ham()
  ei = ham.find('.//pairpot[@name="ElecIon"]')
  ham.remove(ei)
  ei1 = etree.Element('pairpot', {
    'type': 'pseudo',
    'name':'PseudoPot',
    'source': 'ion0',
    'wavefunction': 'psi0',
    'format': 'xml'
  })
  pseudo = etree.Element('pseudo', {
    'elementType': name,
    'href': psp
  })
  ei1.append(pseudo)
  ham.append(ei1)
  return ham


def dmc_sections(tsl, dt=0.4, nwalker=576):
  """ build a list of <qmc method="dmc"> nodes for the given timestep list

  Args:
    tsl (list): a list of timesteps (floats)
    dt (float, optional): imaginary time per block dt = timestep*steps in a.u.
  Return:
    list: a list of etree.Elements containing <qmc method="dmc">
  """
  dmc = xml_examples.pbyp_dmc()

  nodel = []
  for ts in tsl:
    nstep = int(round( np.ceil(dt/ts) ))
    dmc1 = deepcopy(dmc)
    xml.set_param(dmc1, 'steps', str(nstep))
    xml.set_param(dmc1, 'timestep', str(ts))
    xml.set_param(dmc1, 'targetwalkers', str(nwalker))
    nodel.append(dmc1)
  return nodel


def li_estimators():
  """ build a list of <estimator> nodes for typical Li calculation """
  mom_est = etree.Element('estimator', {
    'type': 'momentum',
    'name': 'nofk'
  })
  sk_est = etree.Element('estimator', {
    'type': 'csk',
    'name': 'csk'
  })
  bins = 256
  gr_est = etree.Element('estimator', {
    'type': 'gofr',
    'name': 'gofr',
    'num_bin': str(bins)
  })
  p_est = etree.Element('pressure', {
    'type': 'Pressure'
  })
  estl = [mom_est, sk_est, gr_est, p_est]
  return estl
