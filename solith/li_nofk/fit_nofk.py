import numpy as np

# Fermi k vector magnitude of homogeneous electron gas (heg) at density rs
heg_kfermi = lambda rs:((9*np.pi)/(4.*rs**3.))**(1./3)

def unfold_inv(kvecs,nkm,nke):
  """ unfold inversion symmetry of n(k) data
  Args:
    kvecs (np.array): k vectors
    nkm   (np.array): n(k) mean
    nke   (np.array): n(k) error
  Returns:
    tuple: (kvecs1,nkm1,nke1) unfolded data
  """


  kvecs1 = np.concatenate([kvecs,-kvecs],axis=0)                                
  nkm1 = np.concatenate([nkm]*2,axis=0)                                         
  nke1 = np.concatenate([nke]*2,axis=0)                                         
                                                                                
  return kvecs1,nkm1,nke1
# end def unfold_inv
