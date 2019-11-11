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
Operations
==========

**Module name:** :mod:`thewalrus.operations`

.. currentmodule:: thewalrus.operations

Functions to construct the Fock representation of a Gaussian operation
represented as Symplectic matrix and complex displacements.


Contains some Gaussian operations and auxiliary functions.

Auxiliary functions
-------------------

.. autosummary::
    n_two_mode_squeezed_vac
    choi_expand
    renormalizer

Operations
---------------

.. autosummary::
    fock_tensor
"""

from itertools import product
import numpy as np
from thewalrus import hafnian_batched
from thewalrus.symplectic import two_mode_squeezing, expand
from thewalrus.quantum import Amat

# There are probably a million ways in which to do this in a better way
def n_two_mode_squeezed_vac(n, r=np.arcsinh(1.0)):
    r"""
    Returns the symplectic matrix associated with two mode squeezing operations by amount r between modes i and i+n for 0<=i<n

    Args:
        n (integer): number of modes
        r (float): squeezing parameter
    Returns:
        (array): the symplectic matrix
    """
    ch = np.cosh(r)*np.identity(n)
    sh = np.sinh(r)*np.identity(n)
    zh = np.zeros([n,n])
    Snet = np.block([[ch,sh,zh,zh],[sh,ch,zh,zh],[zh,zh,ch,-sh],[zh,zh,-sh,ch]])
    return Snet

def choi_expand(S, r=np.arcsinh(1.0)):
    r"""
    Construct the Gaussian-Choi-Jamiolkowski expansion of a symplectic matrix S

    Args:
        S (array): symplectic matrix

    Returns:
        (array): expanded symplectic matrix of twice the size of S
    """

    (n, m) = S.shape
    assert n == m
    assert n % 2 == 0
    nmodes = n // 2
    return expand(S, list(range(nmodes)), n) @ n_two_mode_squeezed_vac(nmodes, r)


def renormalizer(tensor, R, l, cutoff):
    r"""
    Returns a renomarlized tensor

    Args:
        tensor (array): Original (unnormarlized) tensor
        R (array): Normalization factors
        l (int): Number of modes
        cutoff (int): Fock space cutoff
    Return:
        (array): The renormalized tensor
    """
    #scaled_tensor = np.empty_like(tensor)
    # Note that the following loops are very inefficient and should be implemented in a better way.
    for p1 in product(list(range(cutoff)), repeat=l):
        for p2 in product(list(range(cutoff)), repeat=l):
            p = tuple(p1 + p2)
            tensor[p] = tensor[p] * np.prod([R[i] for i in p2])
    return tensor


def fock_tensor(S, alpha, cutoff, r=np.arcsinh(1.0)):
    r"""
    Calculated the Fock representation of Gaussian unitary specified by the
    unitary matrices U and Up, the squeezing parameters ls and displacements alpha
    for l modes up to cutoff
    Args:
        S (array): symplectic matrix
        alpha (array): complex vector of displacements
        r (float): squeezing parameter used for the Choi expansion
    Return:
        (array): Tensor containing the Fock representation of the Gaussian unitary
    """
    n, _ = S.shape

    S_exp = choi_expand(S, r)
    cov = S_exp @ S_exp.T
    A = Amat(cov)
    n, _ = A.shape
    N = n // 2
    # B = -A[0:N,0:N].conj()
    B = A[0:N, 0:N].conj()
    alphat = np.array(list(alpha) + list(np.zeros_like(alpha)))
    zeta = alphat - B @ alphat.conj()
    pref = -0.5 * alphat.conj() @ zeta
    R = [1.0 / np.prod((np.tanh(r) ** i) / np.cosh(r)) for i in range(cutoff)]
    # pylint: disable=assignment-from-no-return
    lt = np.arctanh(np.linalg.svd(B, compute_uv=False))
    T = np.exp(pref) / (np.sqrt(np.prod(np.cosh(lt))))

    l = len(alpha)
    tensor = hafnian_batched(
        B, cutoff, mu=zeta, renorm=True
    )  # This is the heavy computational part
    return renormalizer(T * tensor, R, l, cutoff)
