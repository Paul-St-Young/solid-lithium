from lxml import etree
from qharv.seed import xml, xml_examples


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
  gr_est = etree.Element('estimator', {
    'type': 'gofr',
    'name': 'gofr',
    'num_bin': 256
  })
  p_est = etree.Element('pressure', {
    'type': 'Pressure'
  })
  estl = [mom_est, sk_est, gr_est, p_est]
  return estl
