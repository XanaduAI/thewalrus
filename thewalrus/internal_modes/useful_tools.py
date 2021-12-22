import numpy as np 
from numba import jit 


jit(nopython=True, cache=True)
def spatial_modes_to_schmidt_modes(spatial_modes, K):
    """
    returns index of schmidt modes corresponding to the give spatial modes. 
    e.g. if there are K=3 schmidt modes and spatial_modes=[0,2] 
    then schmidt_modes=[0,1,2,6,7,8]

    Args:
        spatial_modes (array): indices of spatial modes
        K (int): number of schmidt modes per spatial mode

    Returns:
        schmidt_modes (array): indices of schmidt modes
    """
    spatial_modes = np.asarray(spatial_modes)
    M = len(spatial_modes)
    schmidt_modes = np.empty(M * K, dtype=spatial_modes.dtype)

    for i in range(K):
        schmidt_modes[i::K] = K * spatial_modes + i

    return schmidt_modes

jit(nopython=True, cache=True)
def spatial_reps_to_schmidt_reps(spatial_reps, K):
    """
    returns reps of schmidt modes corresponding to the give spatial reps. 
    e.g. if there are K=3 schmidt modes and spatial_reps=[1,2] 
    then schmidt_reps=[1,1,1,2,2,2]

    Args:
        spatial_reps (array): number of spatial reps
        K (int): number of schmidt modes per spatial mode

    Returns:
        array: reps of schmidt modes
    """

    M = len(spatial_reps)
    schmidt_reps = np.empty(M * K, dtype=spatial_reps.dtype)
    for i, r in enumerate(spatial_reps):
        schmidt_reps[i*K:(i+1)*K] = r

    return schmidt_reps