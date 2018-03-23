import numpy as np

def nofk_all_twists(df0,sel,obs='nofk'
    ,kmag_cond=None,nkm_cond=None):                      
  """ collect momentum distribution data from all twists                        
  Args:                                                                         
    df (pd.DataFrame): dataframe containing ['kvecs','nkm','nke']               
    sel (np.array): boolean selector array for df                               
    kmag_cond (func1d, optional): conditions on k-vector magnitude              
      e.g. lambda x:x>0.6  # above Fermi surface                                
    nkm_cond (func1d, optional): conditions on n(k) value                       
      e.g. lambda x:x>0.12  # exclude low n(k)                                  
  Returns:                                                                      
    tuple: (points,values,errors) 3D momentum distribution                      
  """                                                                           
                                                                                
  df = df0.loc[sel]                                                             
  ntwist = len(df)  # !!!! assume df contains all twists from a single run      
  cols = ['kvecs',obs+'_mean',obs+'_error']
                                                                                
  kvecl = []                                                                    
  nkml  = []                                                                    
  nkel  = []                                                                    
  for itwist in range(ntwist):                                                  
    kvecs,nkm,nke = df.iloc[itwist][cols].values               
                                                                                
    # select k vectors                                                          
    ksel = np.ones(len(nkm),dtype=bool)                                         
    if kmag_cond is not None:                                                   
      kmags= np.linalg.norm(kvecs,axis=1)                                       
      ksel = kmag_cond(kmags)                                                   
    # end if                                                                    
                                                                                
    # select n(k) values                                                        
    nsel = np.ones(len(nkm),dtype=bool)                                         
    if nkm_cond is not None:                                                    
      nsel = nkm_cond(nkm)                                                      
    # end if                                                                    
                                                                                
    sel1 = ksel&nsel                                                            
    kvecs = np.array(kvecs)[sel1]                                               
    nkm   = np.array(nkm)[sel1]                                                 
    nke   = np.array(nke)[sel1]                                                 
                                                                                
    kvecl.append(kvecs)                                                         
    nkml.append(nkm)                                                            
    nkel.append(nke)                                                            
  # end for                                                                     
                                                                                
  points = np.concatenate(kvecl,axis=0)                                         
  values = np.concatenate(nkml,axis=0)                                          
  errors = np.concatenate(nkel,axis=0)                                          
  return points,values,errors                                                   
# end def nofk_all_twists
