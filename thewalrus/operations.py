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
represented as a Symplectic matrix and complex displacements.


Contains some Gaussian operations and auxiliary functions.

Auxiliary functions
-------------------

.. autosummary::
    n_two_mode_squeezed_vac
    choi_expand

Operations
---------------

.. autosummary::
    fock_tensor
"""

from itertools import product

import numpy as np

from thewalrus import hafnian_batched
from thewalrus.symplectic import expand
from thewalrus.quantum import Amat


def n_two_mode_squeezed_vac(n, r=np.arcsinh(1.0)):
    r"""
    Returns the symplectic matrix associated with two mode squeezing operations.

    Args:
        n (integer): number of modes
        r (float): squeezing parameter
    Returns:
        (array): the symplectic matrix
    """
    ch = np.cosh(r) * np.identity(n)
    sh = np.sinh(r) * np.identity(n)
    zh = np.zeros([n, n])
    Snet = np.block([[ch, sh, zh, zh], [sh, ch, zh, zh], [zh, zh, ch, -sh], [zh, zh, -sh, ch]])
    return Snet


def choi_expand(S, r=np.arcsinh(1.0)):
    r"""
    Construct the Gaussian-Choi-Jamiolkowski expansion of a symplectic matrix S.

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


def fock_tensor(S, alpha, cutoff, r=np.arcsinh(1.0)):
    r"""
    Calculates the Fock representation of a Gaussian unitary.
    and displacements alpha up to cutoff
    Args:
        S (array): symplectic matrix
        alpha (array): complex vector of displacements
        cutoff (int): cutoff in Fock space
        r (float): squeezing parameter used for the Choi expansion
    Return:
        (array): Tensor containing the Fock representation of the Gaussian unitary
    """
    S_exp = choi_expand(S, r)
    cov = S_exp @ S_exp.T
    A = Amat(cov)
    n, _ = A.shape
    N = n // 2
    B = A[0:N, 0:N].conj()
    l = len(alpha)
    alphat = np.array(list(alpha) + ([0] * l))
    zeta = alphat - B @ alphat.conj()
    pref_exp = -0.5 * alphat.conj() @ zeta
    R = [1.0 / np.prod((np.tanh(r) ** i) / np.cosh(r)) for i in range(cutoff)]
    # pylint: disable=assignment-from-no-return
    lt = np.arctanh(np.linalg.svd(B, compute_uv=False))
    T = np.exp(pref_exp) / (np.sqrt(np.prod(np.cosh(lt))))

    tensor = T * hafnian_batched(
        B, cutoff, mu=zeta, renorm=True
    )  # This is the heavy computational part
    vals = list(range(l))
    vals2 = list(range(l, 2 * l))
    tensor_view = tensor.transpose(vals2 + vals)
    # There is probably a better way to do the following rescaling, but this is already "good"
    for p in product(list(range(cutoff)), repeat=l):
        tensor_view[p] = tensor_view[p] * np.prod([R[i] for i in p])
    return tensor
