# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Hafnian and torontonian sampling
"""
# pylint: disable=too-many-arguments
import random

import numpy as np

from .quantum import Amat, Beta, Qmat, prefactor, density_matrix_element, reduced_gaussian, kron_reduced
from .lib.libtor import torontonian_samples as libtor
from hafnian.lib.libhaf import torontonian_complex
import hafnian 




def generate_hafnian_sample(cov, hbar=2, cutoff=6):
    r"""Returns a single sample from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        mu (array): a length-:math:`2N` ``np.float64`` vector of means.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cutoff (int): the Fock basis truncation (optional). This overwrites
            ``tol`` if used.
        tol (float): determines dynamically the cutoff to use, by determining
            :math:`D` such that :math:`\sum_{i=0}^D\mathfrak{P}_i > 1-\epsilon`.

    Returns:
        np.array[int]: samples from the Hafnian of the Gaussian state.
    """
    N = len(cov)//2
    result = []
    prev_prob = 1.0
    nmodes = N
    mu = np.zeros(2*N)

    for k in range(nmodes):
        probs1 = np.zeros([cutoff+1], dtype=np.float64)
        kk = np.arange(k+1)
        mu_red, V_red = reduced_gaussian(mu, cov, kk)
        Q = Qmat(V_red, hbar=hbar)
        A = Amat(Q, hbar=hbar, cov_is_qmat=True)
     
        for i in range(cutoff):
            indices = result+[i]
            ind2 = indices+indices
            factpref = np.prod([np.math.factorial(l) for l in indices])

            probs1[i] = hafnian.hafnian(kron_reduced(A,ind2)).real/factpref
        probs1a = probs1/np.sqrt(np.linalg.det(Q).real)
        probs2 = (probs1a)/prev_prob
        probs3 = np.maximum(probs2,np.zeros_like(probs2))
        ssum = np.sum(probs3)
        if ssum < 1.0:
            probs3[-1] = 1.0-ssum
        #if np.isnan(np.min(probs3)):
        #    result = -1*np.ones(nmodes, dtype=np.int16)
        #    return result.tolist()
        
        result.append(np.random.choice(a=range(len(probs3)), p=probs3))
        if result[-1] == cutoff:
            return -1


        prev_prob = probs1a[result[-1]]

#       if np.sum(result)>30:
#           break
#       print(k,prev_prob,np.sum(result))
    return result




def hafnian_sample(cov, samples=1, hbar=2, cutoff=5):
    r"""Returns samples from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        mu (array): a length-:math:`2N` ``np.float64`` vector of means.
        samples (int): the number of samples to return.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cutoff (int): the Fock basis truncation (optional). This overwrites
            ``tol`` if used.
        tol (float): determines dynamically the cutoff to use, by determining
            :math:`D` such that :math:`\sum_{i=0}^D\mathfrak{P}_i > 1-\epsilon`.

    Returns:
        np.array[int]: samples from the Torontonian of the covariance matrix.
    """
    if not isinstance(cov, np.ndarray):
        raise TypeError("Covariance matrix must be a NumPy array.")

    matshape = cov.shape
    N = matshape[0]

    if matshape[0] != matshape[1]:
        raise ValueError("Covariance matrix must be square.")

    if np.isnan(cov).any():
        raise ValueError("Covariance matrix must not contain NaNs.")

    samples_array = []
    j=0
    #for _ in range(samples):
    while j < samples:
        result = generate_hafnian_sample(cov, hbar=hbar, cutoff=cutoff)
        if result != -1:
            samples_array.append(result)
            j = j+1
        # The if above implies that you never get see anything beyond cutoff
    return np.vstack(samples_array)
    #return samples_array


def torontonian_sample(cov, samples=1):
    r"""Returns samples from the Torontonian of a Gaussian state 

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.

    Returns:
        np.array[int]: samples from the Torontonian of the covariance matrix.
    """
    if not isinstance(cov, np.ndarray):
        raise TypeError("Covariance matrix must be a NumPy array.")

    matshape = cov.shape
    N = matshape[0]

    if matshape[0] != matshape[1]:
        raise ValueError("Covariance matrix must be square.")

    if np.isnan(cov).any():
        raise ValueError("Covariance matrix must not contain NaNs.")



    samples_array = []
    for _ in range(samples):
        seed = random.randint(0, 10**6)
        samples_array.append(generate_torontonian_sample(cov))

    return np.vstack(samples_array)

def Xmat(nmodes):
    X = np.block([[0*np.identity(nmodes),np.identity(nmodes)],[np.identity(nmodes),0*np.identity(nmodes)]])
    return X


def tor(O):
    return torontonian_complex(O,quad=False).real

def generate_torontonian_sample(cov, hbar=2):
    r"""Returns a single sample from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        np.array[int]: samples from the Hafnian of the Gaussian state.
    """
    result=[]
    (n1,n2) = cov.shape
    assert n1==n2
    nmodes=n1//2
    prev_prob=1.0
    mu = np.zeros(n1)
    for k in range(nmodes):
        probs1 = np.zeros([2], dtype=np.float64)
        kk = np.arange(k+1)
        mu_red, V_red = reduced_gaussian(mu, cov, kk)
        Q = Qmat(V_red, hbar=hbar)
        A = Amat(Q, hbar=hbar, cov_is_qmat=True)

        O = Xmat(k+1) @ A

        indices = result+[0]
        ind2 = indices+indices    

        probs1[0] = tor(np.complex128(kron_reduced(O,ind2))).real

        indices = result+[1]
        ind2 = indices+indices    
        probs1[1] = tor(np.complex128(kron_reduced(O,ind2))).real

        probs1a = probs1/np.sqrt(np.linalg.det(Q).real)
        probs2=(probs1a)/prev_prob
        probs3 = np.maximum(probs2,np.zeros_like(probs2))
        #ssum = np.sum(probs3)
        #if ssum < 1.0:
        #    probs3[-1] = 1.0-ssum
        #print(probs3)
        probs3/=np.sum(probs3)
        result.append(np.random.choice(a=range(len(probs3)), p=probs3))

        prev_prob = probs1a[result[-1]]
        if np.sum(result)>=30:
            break
    return result