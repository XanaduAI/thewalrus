import numpy as np 
import numba
from numba import prange
from thewalrus._hafnian import (
    precompute_binoms,
    nb_ix,
    matched_reps,
    find_kept_edges,
    f_loop,
    f_loop_odd,
    get_submatrices,
    get_submatrix_batch_odd0,
    get_Dsubmatrices,
    eigvals
    )
from loop_hafnian_batch import add_batch_edges_odd, add_batch_edges_even

@numba.jit(nopython=True, cache=True, parallel=True)
def _calc_loop_hafnian_batch_gamma_even(A, D, fixed_edge_reps,
        batch_max, odd_cutoff,
        glynn=True):

    oddloop = D[:,0]
    oddV = A[0,:]

    n = A.shape[0]
    N_fixed = 2 * fixed_edge_reps.sum() # number of photons

    N_max = N_fixed + 2 * batch_max + odd_cutoff

    edge_reps = np.concatenate((np.array([batch_max]), fixed_edge_reps))
    steps = np.prod(edge_reps + 1)
    # precompute binomial coefficients 
    max_binom = edge_reps.max() + odd_cutoff
    binoms = precompute_binoms(max_binom)
    n_D=D.shape[0]

    H_batch = np.zeros((n_D,2*batch_max+odd_cutoff+1), dtype=np.complex128)

    for j in prange(steps):
        
        Hnew=np.zeros((n_D,2*batch_max+odd_cutoff+1), dtype=np.complex128)

        kept_edges = find_kept_edges(j, edge_reps)
        edges_sum = kept_edges.sum()

        binom_prod = 1.
        for i in range(1, n//2):
            binom_prod *= binoms[edge_reps[i], kept_edges[i]]
        
        if glynn:
            delta = 2 * kept_edges - edge_reps
        else:
            delta = kept_edges

        AX_S, XD_S, D_S, oddVX_S = get_submatrices(delta, A, D[0,:], oddV)

        AX_S_copy = AX_S.copy()
        
        for k in range(n_D):
            XD_S,D_S=get_Dsubmatrices(delta,D[k,:])
            
            f_even = f_loop(AX_S_copy, AX_S, XD_S, D_S, N_max)
            f_odd = f_loop_odd(AX_S_copy, AX_S, XD_S, D_S, N_max, oddloop[k], oddVX_S)
    
            for N_det in range(2*kept_edges[0], 2*batch_max+odd_cutoff+1):
                N = N_fixed + N_det
                plus_minus = (-1.) ** (N // 2 - edges_sum)
    
                n_det_binom_prod = binoms[N_det//2, kept_edges[0]] * binom_prod
    
                if N_det % 2 == 0:
                    Hnew[k,N_det] += n_det_binom_prod * plus_minus * f_even[N//2]
                else: 
                    Hnew[k,N_det] += n_det_binom_prod * plus_minus * f_odd[N]
        H_batch+=Hnew
        
    if glynn:
        for j in range(H_batch.shape[1]):
            x=N_fixed+j
            H_batch[:,j]*=0.5**(x//2)

    return H_batch

@numba.jit(nopython=True, cache=True, parallel=True)
def _calc_loop_hafnian_batch_gamma_odd(A, D, fixed_edge_reps,
        batch_max, even_cutoff, oddmode,
        glynn=True):

    oddloop = D[:,0]
    oddV = A[0,:]

    #when I added the extra edges, I place the edge which goes from the oddmode to 
    #to the current mode in the index 1 position of the array
    oddloop0 = D[:,1]
    oddV0 = A[1,:]

    n = A.shape[0]
    N_fixed = 2 * fixed_edge_reps.sum() + 1
    N_max = N_fixed + 2 * batch_max + even_cutoff + 1
    
    n_D=D.shape[0]

    edge_reps = np.concatenate((np.array([batch_max, 1]), fixed_edge_reps))
    steps = np.prod(edge_reps + 1)
    # precompute binomial coefficients 
    max_binom = edge_reps.max() + even_cutoff
    binoms = precompute_binoms(max_binom)

    H_batch = np.zeros((n_D,2*batch_max+even_cutoff+2), dtype=np.complex128)
    #for j in range(rank, steps, size):
    for j in prange(steps):
        
        Hnew=np.zeros((n_D,2*batch_max+even_cutoff+2), dtype=np.complex128)
        
        kept_edges = find_kept_edges(j, edge_reps)
        edges_sum = kept_edges.sum()

        binom_prod = 1.
        for i in range(1, n//2):
            binom_prod *= binoms[edge_reps[i], kept_edges[i]]
        
        if glynn:
            delta = 2 * kept_edges - edge_reps
        else:
            delta = kept_edges

        AX_S, XD_S, D_S, oddVX_S = get_submatrices(delta, A, D[0,:], oddV)

        AX_S_copy = AX_S.copy()

        for k in range(n_D):
            XD_S,D_S=get_Dsubmatrices(delta,D[k,:])
            
            if kept_edges[0] == 0 and kept_edges[1] == 0:
                oddVX_S0 = get_submatrix_batch_odd0(delta, oddV0)
                plus_minus = (-1) ** (N_fixed // 2 - edges_sum)
                f = f_loop_odd(AX_S_copy, AX_S, XD_S, D_S, N_fixed, oddloop0[k], oddVX_S0)[N_fixed]
                Hnew[k,0] += binom_prod * plus_minus * f 
    
            f_even = f_loop(AX_S_copy, AX_S, XD_S, D_S, N_max)
            f_odd = f_loop_odd(AX_S_copy, AX_S, XD_S, D_S, N_max, oddloop[k], oddVX_S)
    
            for N_det in range(2*kept_edges[0]+1, 2*batch_max+even_cutoff+2):
                N = N_fixed + N_det
                plus_minus = (-1) ** (N // 2 - edges_sum)
    
                n_det_binom_prod = binoms[(N_det-1)//2, kept_edges[0]] * binom_prod
    
                if N % 2 == 0:
                    Hnew[k,N_det] += n_det_binom_prod * plus_minus * f_even[N//2]
                else: 
                    Hnew[k,N_det] += n_det_binom_prod * plus_minus * f_odd[N]
        
        H_batch+=Hnew
                    
    if glynn:
        for j in range(H_batch.shape[1]):
            x=N_fixed+j
            H_batch[:,j]*=0.5**(x//2)

    return H_batch

def loop_hafnian_batch_gamma(A, D, fixed_reps, N_cutoff, glynn=True):
    # checks 
    n = A.shape[0]
    assert A.shape[1] == n
    assert D.shape[1] == n
    assert len(fixed_reps) == n - 1 

    nz = np.nonzero(list(fixed_reps) + [1])[0]
    Anz = A[np.ix_(nz, nz)]
    Dnz = D[:,nz]

    fixed_reps = np.asarray(fixed_reps)
    fixed_reps_nz = fixed_reps[nz[:-1]]

    fixed_edges, fixed_m_reps, oddmode = matched_reps(fixed_reps_nz)

    if oddmode is None:
        batch_max = N_cutoff // 2
        odd_cutoff = N_cutoff % 2
        edges = add_batch_edges_even(fixed_edges)
        Ax = Anz[np.ix_(edges, edges)].astype(np.complex128)
        Dx = Dnz[:,edges].astype(np.complex128)
        return _calc_loop_hafnian_batch_gamma_even(Ax, Dx, fixed_m_reps, batch_max, odd_cutoff,
            glynn=glynn)
    else:
        edges = add_batch_edges_odd(fixed_edges, oddmode)
        Ax = Anz[np.ix_(edges, edges)].astype(np.complex128)
        Dx = Dnz[:,edges].astype(np.complex128)
        batch_max = (N_cutoff-1) // 2
        even_cutoff = 1 - (N_cutoff % 2)
        return _calc_loop_hafnian_batch_gamma_odd(Ax, Dx, fixed_m_reps, batch_max, even_cutoff, oddmode,
            glynn=glynn)
