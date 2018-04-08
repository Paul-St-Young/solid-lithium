import numpy as np
from qharv.plantation import sugar

# Fermi k vector magnitude of homogeneous electron gas (heg) at density rs
heg_kfermi = lambda rs:((9*np.pi)/(4.*rs**3.))**(1./3)


def unfold_inv(kvecs,nkm,nke):
  """ unfold inversion symmetry of n(k) data
  Args:
    kvecs (np.array): k vectors
    nkm   (np.array): n(k) mean
    nke   (np.array): n(k) error
  Return:
    tuple: (kvecs1,nkm1,nke1) unfolded data
  """
  kvecs1 = np.concatenate([kvecs,-kvecs],axis=0)                                
  nkm1 = np.concatenate([nkm]*2,axis=0)                                         
  nke1 = np.concatenate([nke]*2,axis=0)                                         
                                                                                
  return kvecs1,nkm1,nke1
# end def unfold_inv


@sugar.check_file_before
def save_bspline(fyaml, tck):
  """ save Bspline coefficients to file, abort if file exists

  example:
    import scipy.interpolate as interp
    tck = interp.splrep(x, y)
    save_bspline('myspline.yml', tck)

  Args:
    fyaml (str): filename to store yaml entry
    tck (tuple): length-3 tuple storing (knots, coeffs, order)
  """
  import yaml
  knots, coeffs, order = tck
  entry = {
    'knots':knots.tolist()
   ,'coeffs':coeffs.tolist()
   ,'order':int(order)
  }
  with open(fyaml,'w') as f:
    yaml.dump(entry,f)
# end def save_bspline


def load_bspline(fyaml):
  """ inverse of save_bspline

  example:
    import scipy.interpolate as interp
    tck = load_bspline('myspline.yml')
    myy = interp.splev(x, tck)
    fit_err = abs(y - myy)

  Args:
    fyaml (str): filename holding yaml entry
  Return
    tuple: length-3 tuple storing (knots, coeffs, order)
  """
  import yaml
  with open(fyaml,'r') as f:
    entry = yaml.load(f)
  knots  = np.array(entry['knots'])
  coeffs = np.array(entry['coeffs'])
  order  = entry['order']

  tck = (knots, coeffs, order)
  return tck
# end def load_bspline
