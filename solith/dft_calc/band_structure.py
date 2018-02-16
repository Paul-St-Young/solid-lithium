import json
import seekpath

def get_explicit_kpaths(seek_json,dk=0.025):
  """ convert implicit kpath info to explicit list of kpoints
  make simple edits to the JSON file returned from the seeK-path website
  so that it would be accepted by seekpath.getpaths.get_explicit_from_implicit

  example of implicit kpath:
    'path':[
      ['GAMMA', 'H'],
      ['H', 'N'],
      ['N', 'GAMMA'],
      ['GAMMA', 'P'],
      ['P', 'H'],
      ['P', 'N']
    ],
    'kpoints_rel':[
      'GAMMA':[0.0,0.0,0.0],
      'H':[0.5,-0.5,0.5],
      'N':[0.0,0.0,0.5],
      'P':[0.25,0.25,0.25]
    ]

  example of explicity kpath:
    'kpoints_labels':['Gamma','',...,'N'],
    'kpoints_abs': array([
      [0.,0.,0.]
      [0.018,0.,0.018],
      ...,
      [1.618,1.618,3.236]
    ])

  Args:
    seek_json (str): JSON file pasted from seeK-path website
    dk (float, optional): spacing in reciprocal space in Hartree a.u.
  Return:
    dict: return value of seekpath.getpaths.get_explicit_from_implicit
  """
  from qharv.inspect import axes_pos  # need raxes
  with open(seek_json,'r') as f:
    data = json.load(f)
  # end with
  data['point_coords'] = data['kpoints']                                        
  data['reciprocal_primitive_lattice'] = axes_pos.raxes(
    data['primitive_lattice']
  )
  kpaths = seekpath.getpaths.get_explicit_from_implicit(data,dk)                
  return kpaths                                                                 
# end def get_explicit_kpaths

def plot_bands(ax,karr,earr,efermi,**kwargs):
  """ draw band structure on ax
  Args:
    ax (matplotlib.Axes): axes to draw on
    karr (np.array): (nband,nk), location along reciprocal space path
    earr (np.array): (nband,nk), band energy
    efermi (float): Fermi energy
  """

  nbnd,nk   = karr.shape
  nbnd1,nk1 = earr.shape
  if nbnd != nbnd1:
    raise RuntimeError('number of bands mismatch')
  if nk != nk1:
    raise RuntimeError('number of k-points mismatch')
  
  ax.axhline(efermi)

  for ibnd in range(nbnd):                                                      
    myx = karr[ibnd]                                                            
    myy = earr[ibnd]                                                            
                                                                                
    has_filled = has_empty = False                                              
    filled_sel = myy < efermi                                                   
    if len(myx[filled_sel]) > 0:                                                
      has_filled = True                                                         
    if len(myx[~filled_sel]) > 0:                                               
      has_empty = True                                                          
                                                                                
    fmt = '-'                                                                   
    if has_filled and has_empty: fmt = '.'                                      
                                                                                
    if has_filled:                                                              
      ax.plot(myx[filled_sel],myy[filled_sel],fmt,c='k',**kwargs)                        
    if has_empty:                                                               
      ax.plot(myx[~filled_sel],myy[~filled_sel],fmt,c='gray')                   
  # end for ibnd
  
# end def plot_bands

