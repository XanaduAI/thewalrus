import numpy as np
import numba


@numba.jit
def hafnian(m):
    nb_lines = len(m)
    nb_columns = len(m[0])
    if nb_lines != nb_columns:
        raise ValueError('Matrix must be square')
    
    if nb_lines % 2 != 0:
        raise ValueError('Matrix size must be even')
    
    n = int(float(len(m))/2)
    z = np.zeros((n*(2*n-1),n+1))
    for j in range(1, 2*n):
        for k in range(j):
            z[int(j*(j-1)/2+k)][0] = m.copy()[j][k]
    g = np.zeros(n+1)
    g[0] = 1
    return solve(z, 2*n, 1, g, n)

@numba.jit
def solve(b,s,w,g,n):
    if s == 0:
        return w*g[n]
    c = np.zeros((int((s-2)*(s-3)/2), n+1))
    i=0
    for j in range(1,s-2):
        for k in range(j):
            c[i] = b[int((j+1)*(j+2)/2+k+2)]
            i+=1
    h = solve(c, s-2, -w, g, n)
    e = g[:].copy()
    for u in range(n):
        for v in range(n-u):
            e[u+v+1] += g[u]*b[0][v]
    for j in range(1, s-2):
        for k in range(j):
            for u in range(n):
                for v in range(n-u):
                    c[int(j*(j-1)/2+k)][u+v+1] += b[int((j+1)*(j+2)/2)][u]*b[int((k+1)*(k+2)/2+1)][v] + b[int((k+1)*(k+2)/2)][u]*b[int((j+1)*(j+2)/2+1)][v]
    return h + solve(c, s-2, w, e, n)
