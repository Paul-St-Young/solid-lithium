# Paola PRB 66, 235116 (2002)
def paola_nm(rs):
  v1 = -0.0679793
  v2 = -0.00102846
  v3 = 0.000189111
  v4 = 0.0205397
  v5 = -0.0086838
  v6 = 6.87109e-5
  v7 = 4.868047e-5
  nume = 1+v1*rs+v2*rs**2+v3*rs**3
  deno = 1+v4*rs+v5*rs**2+v6*rs**3+v7*rs**(15./4)
  return nume/deno
def paola_np(rs):
  q1 = 0.088519
  q2 = 0.45
  q3 = 0.022786335
  nume = q1*rs
  deno = 1+q2*rs**0.5+q3*rs**(7./4)
  return nume/deno
def paola_zkf(rs):
  nminus = paola_nm(rs)
  nplus = paola_np(rs)
  return nminus-nplus
