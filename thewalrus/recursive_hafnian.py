

import numpy as np
import numba

@numba.jit
def hafnian(m):
  n = len(m)/2
  z = [[0]*(n+1) for _ in range(n*(2*n-1))]
  for j in range(1, 2*n):
    for k in range(j):
      z[j*(j-1)/2+k][0] = m[j][k]
  return solve(z, 2*n, 1, [1] + [0]*n, n)

@numba.jit
def solve(b, s, w, g, n):
  if s == 0:
    return w*g[n]
  c = [b[(j+1)*(j+2)/2+k+2][:] for j in range(1, s-2) for k in range(j)]
  h = solve(c, s-2, -w, g, n)
  e = g[:]
  for u in range(n):
    for v in range(n-u):
      e[u+v+1] += g[u]*b[0][v]
  for j in range(1, s-2):
    for k in range(j):
      for u in range(n):
        for v in range(n-u):
          c[j*(j-1)/2+k][u+v+1] += b[(j+1)*(j+2)/2][u]*b[(k+1)*(k+2)/2+1][v] + b[(k+1)*(k+2)/2][u]*b[(j+1)*(j+2)/2+1][v]
  return h + solve(c, s-2, w, e, n)
