import numba 
import numpy as np 

@numba.jit(nopython=True, cache=True)
def nb_binom(n, k):
    """
    numba version of binomial coefficient function
    Args:
        n (int): how many options
        k (int): how many are chosen
    Returns:
        int: how many ways of choosing
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    binom = 1
    for i in range(min(k, n - k)):
        binom *= (n - i)
        binom //= (i + 1)
    return binom

@numba.jit(nopython=True, cache=True)
def precompute_binoms(max_binom):
    """
    precompute binomial coefficients, return as a 2d array
    Args:
        max_binom (int): max value of n in the binomial
    Returns:
        array: max_binom+1 x max_binom+1 array of binomial coefficients
    """
    binoms = np.zeros((max_binom+1, max_binom+1), dtype=type(max_binom))
    for i in range(max_binom+1):
        for j in range(max_binom+1):
            binoms[i,j] = nb_binom(i,j)
    return binoms

@numba.jit(nopython=True, cache=True)
def nb_ix(arr, rows, cols):
    """
    numba implementation of np.ix_
    Args:
        arr (2d array) : matrix to take submatrix of
        rows (array) : rows to be selected in submatrix
        cols (array) : columns to be selected in submatrix
    Return: 
        len(rows) * len(cols) array : selected submatrix of arr
    """
    return arr[rows][:, cols]

def matched_reps(reps):
    """
    takes the repeated rows and find a way to pair them up
    to create a perfect matching with many repeated edges
    Args:
        reps (list): list of repeated rows/cols
    Returns:
        x (array): vertex pairs, length 2N for N edges. Index i is matched with i+N
        edgereps (array): length N array for how many times each edge is repeated
        oddmode (int or None): index of odd mode, None if even number of vertices
    """
    n = len(reps)

    if sum(reps) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), None

    # need to pair off the indices with high numbers of repetitions...
    x = range(n) # the starting set of indices
    edgesA = []  # contains part A of each pair
    edgesB = [] # part B of each pair
    edgereps = [] # number of repetitions of a pair
    reps, x = zip(*sorted(zip(reps, x), reverse=True)) # sort according to reps, in descending order
    reps = list(reps)
    x = list(x)

    # remove zeros
    nonzero_reps = []
    nonzero_x = []
    for i, r in zip(x, reps):
        if r > 0:
            nonzero_reps.append(r)
            nonzero_x.append(i)
    reps = nonzero_reps
    x = nonzero_x

    while len(reps) > 1 or (len(reps) == 1 and reps[0] > 1):
        reps, x = zip(*sorted(zip(reps, x), reverse=True)) #sort
        reps = list(reps)
        x = list(x)
        if len(reps) == 1 or reps[0] > reps[1] * 2:
            #if largest number of reps is more than double the 2nd largest, pair it with itself
            edgesA += [x[0]]
            edgesB += [x[0]]
            edgereps += [reps[0]//2]
            if reps[0] % 2 == 0:
                x = x[1:]
                reps = reps[1:]
            else:
                reps[0] = 1
        else:
            #otherwise, form pairs between largest reps and 2nd largest reps
            edgesA += [x[0]]
            edgesB += [x[1]]
            edgereps += [reps[1]]
            if reps[0] > reps[1]:
                if len(x) > 2:
                    x = [x[0]] + x[2:]
                    reps = [reps[0]-reps[1]] + reps[2:]
                else:
                    x = [x[0]]
                    reps = [reps[0]-reps[1]]
            else:
                x = x[2:]
                reps = reps[2:]
                
    if len(x)==1:
        oddmode=x[0] #if there is an unpaired mode, store it
    else:
        oddmode=None

    # the adjacency matrix of red edges connects 1 to N/2+1, 2 to N/2+2, etc.
    # Reorder the indices (from x2 back to x) so that the paired indices get
    # connected by red edges
    x = np.asarray(edgesA + edgesB, dtype=np.int64) #reordered list of indices
    edgereps = np.asarray(edgereps, dtype=np.int64)

    return x, edgereps, oddmode

@numba.jit(nopython=True, cache=True)
def find_kept_edges(j, reps):
    """
    write j as a string where the ith digit is in base (reps[i]+1)
    decides which edges are included given index of the inclusion/exclusion sum
    Args:
        j (int): index of sum
        reps (list): number of repetitions of each edge
    Returns:
        array : number of repetitions kept for the current inclusion/exclusion step
    """
    num = j
    output = []
    bases = np.asarray(reps) + 1
    for base in bases[::-1]:
        output.append(num % base)
        num //= base 
    return np.array(output[::-1], dtype=reps.dtype)

@numba.jit(nopython=True, cache=True)
def f(E, n):
    """
    evaluate the polyonial coefficients of the function in the eigevalue-trace formula
    Args:
        E (array): eigevavlues of AX
        n (int): number of polynomial coefficients to compute
    Returns:
        array: polynomial coefficients
    """
    E_k = E.copy()
    # Compute combinations in O(n^2log n) time 
    # code translated from thewalrus matlab script
    count = 0
    comb = np.zeros((2, n//2+1), dtype=np.complex128)    
    comb[0,0] = 1
    for i in range(1, n//2+1):
        factor = E_k.sum() / (2 * i)
        E_k *= E
        powfactor = 1
        count = 1 - count
        comb[count, :] = comb[1-count, :]
        for j in range(1, n//(2*i)+1):
            powfactor *= factor / j
            for k in range(i*j+1, n//2+2):
                comb[count, k-1] += comb[1-count, k-i*j-1] * powfactor        
    return comb[count, :]

@numba.jit(nopython=True, cache=True)
def f_loop(E, AX_S, XD_S, D_S, n):
    """
    evaluate the polyonial coefficients of the function in the eigevalue-trace formula
    Args:
        E (array): eigevavlues of AX
        AX_S (array): AX_S with weights given by repetitions and exluded rows removed
        XD_S (array): Diagonal multiplied by X
        D_S (array): Diagonal
        n (int): number of polynomial coefficients to compute
    Returns:
        array: polynomial coefficients
    """
    E_k = E.copy()
    # Compute combinations in O(n^2log n) time 
    # code translated from thewalrus matlab script
    count = 0
    comb = np.zeros((2, n//2+1), dtype=np.complex128)    
    comb[0,0] = 1
    for i in range(1, n//2+1):
        factor = E_k.sum() / (2 * i) + (XD_S @ D_S) / 2
        E_k *= E
        XD_S = XD_S @ AX_S
        powfactor = 1
        count = 1 - count
        comb[count, :] = comb[1-count, :]
        for j in range(1, n//(2*i)+1):
            powfactor *= factor / j
            for k in range(i*j+1, n//2+2):
                comb[count, k-1] += comb[1-count, k-i*j-1] * powfactor        
    return comb[count, :]

@numba.jit(nopython=True, cache=True)
def f_loop_odd(E, AX_S, XD_S, D_S, n, oddloop, oddVX_S):
    """
    evaluate the polyonial coefficients of the function in the eigevalue-trace formula
    when there is a self-edge in the fixed perfect matching
    Args:
        E (array): eigevavlues of AX
        AX_S (array): AX_S with weights given by repetitions and exluded rows removed
        XD_S (array): Diagonal multiplied by X
        D_S (array): Diagonal
        n (int): number of polynomial coefficients to compute
        oddloop (float): weight of self-edge
        oddVX_S (array): vector corresponding to matrix at the index of the self-edge
    Returns:
        array: polynomial coefficients
    """
    E_k = E.copy()

    count = 0
    comb = np.zeros((2, n+1), dtype=np.complex128)    
    comb[0,0] = 1
    for i in range(1, n+1):
        if i==1:
            factor = oddloop
        elif i%2==0:
            factor = E_k.sum() / i + (XD_S @ D_S) / 2
            E_k *= E
        else:
            factor = oddVX_S @ D_S
            D_S = AX_S @ D_S

        powfactor = 1
        count = 1 - count
        comb[count, :] = comb[1-count, :]
        for j in range(1, n//i+1):
            powfactor *= factor / j
            for k in range(i*j+1, n+2):
                comb[count, k-1] += comb[1-count, k-i*j-1] * powfactor        

    return comb[count, :]

@numba.jit(nopython=True, cache=True)
def get_AX_S(kept_edges, A):
    """
    given the kept edges, return the appropriate scaled submatrices to compute f
    Args:
        kept_edges (array): number of repetitions of each edge
        A (array): matrix before repetitions applied
    Returns:
        AX_nonzero (array): scaled A @ X (where X = ((0, I), (I, 0))
    """

    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]

    A_nonzero = nb_ix(A, nonzero_rows, nonzero_rows)

    AX_nonzero = np.empty_like(A_nonzero, dtype=np.complex128)
    AX_nonzero[:,:n_nonzero_edges] = kept_edges_nonzero * A_nonzero[:,n_nonzero_edges:]
    AX_nonzero[:,n_nonzero_edges:] = kept_edges_nonzero * A_nonzero[:,:n_nonzero_edges]

    return AX_nonzero

@numba.jit(nopython=True, cache=True)
def get_submatrices(kept_edges, A, D, oddV):
    """
    given the kept edges, return the appropriate scaled submatrices to compute f
    Args:
        kept_edges (array): number of repetitions of each edge
        A (array): matrix before repetitions applied
        D (array): diagonal before repetitions applied 
        oddV (array): row of matrix at index of self-edge, None is no self-edge
    Returns:
        AX_nonzero (array): scaled A @ X (where X = ((0, I), (I, 0))
        XD_nonzero (array): scaled X @ D 
        D_nonzero (array): scaled D
        oddVX_nonzero (array): scaled oddV @ X
    """

    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]

    A_nonzero = nb_ix(A, nonzero_rows, nonzero_rows)

    AX_nonzero = np.empty_like(A_nonzero, dtype=np.complex128)
    AX_nonzero[:,:n_nonzero_edges] = kept_edges_nonzero * A_nonzero[:,n_nonzero_edges:]
    AX_nonzero[:,n_nonzero_edges:] = kept_edges_nonzero * A_nonzero[:,:n_nonzero_edges]

    D_nonzero = D[nonzero_rows]

    XD_nonzero = np.empty_like(D_nonzero, dtype=np.complex128)
    XD_nonzero[:n_nonzero_edges] = kept_edges_nonzero * D_nonzero[n_nonzero_edges:]
    XD_nonzero[n_nonzero_edges:] = kept_edges_nonzero * D_nonzero[:n_nonzero_edges]

    if oddV is not None:
        oddV_nonzero = oddV[nonzero_rows]
        oddVX_nonzero = np.empty_like(oddV_nonzero, dtype=np.complex128)
        oddVX_nonzero[:n_nonzero_edges] = kept_edges_nonzero * oddV_nonzero[n_nonzero_edges:]
        oddVX_nonzero[n_nonzero_edges:] = kept_edges_nonzero * oddV_nonzero[:n_nonzero_edges]
    else:
        oddVX_nonzero = None 

    return AX_nonzero, XD_nonzero, D_nonzero, oddVX_nonzero

@numba.jit(nopython=True, cache=True)
def get_submatrix_batch_odd0(kept_edges, oddV0):
    """
    find oddVX_nonzero0 for batching (sometimes different vertices are identified as self edges)
    Args:
        kept_edges (array): number of repetitions of each edge
        oddV0 (array): row of matrix at index of self-edge, None is no self-edge
    
    Returns:
        array: scaled oddV0 @ X for 
    """
    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]
    oddV_nonzero0 = oddV0[nonzero_rows]
    oddVX_nonzero0 = np.empty_like(oddV_nonzero0, dtype=np.complex128)
    oddVX_nonzero0[:n_nonzero_edges] = kept_edges_nonzero * oddV_nonzero0[n_nonzero_edges:]
    oddVX_nonzero0[n_nonzero_edges:] = kept_edges_nonzero * oddV_nonzero0[:n_nonzero_edges]

    return oddVX_nonzero0

@numba.jit(nopython=True, cache=True)
def get_Dsubmatrices(kept_edges, D):
    """
    find submatrices for batch gamma functions
    """

    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]

    D_nonzero = D[nonzero_rows]

    XD_nonzero = np.empty_like(D_nonzero, dtype=np.complex128)
    XD_nonzero[:n_nonzero_edges] = kept_edges_nonzero * D_nonzero[n_nonzero_edges:]
    XD_nonzero[n_nonzero_edges:] = kept_edges_nonzero * D_nonzero[:n_nonzero_edges]

    return XD_nonzero, D_nonzero

@numba.jit(nopython=True, cache=True)
def eigvals(M):
    return np.linalg.eigvals(M)

# @numba.jit(nopython=True, cache=True)
def calc_approx_steps(fixed_reps, N_cutoff):
    steps = int(np.prod(np.sqrt(fixed_reps))+1) * N_cutoff // 2
    return steps