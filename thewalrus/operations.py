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
    choi_expand

Operations
---------------

.. autosummary::
    fock_tensor
"""

from itertools import product

import numpy as np

from thewalrus import hafnian_batched
from thewalrus.symplectic import expand, is_symplectic
from thewalrus.quantum import Amat


def choi_expand(S, r=np.arcsinh(1.0)):
    r"""
    Construct the Gaussian-Choi-Jamiolkowski expansion of a symplectic matrix S.

    Args:
        S (array): symplectic matrix

    Returns:
        (array): expanded symplectic matrix of twice the size of S
    """

    (n, m) = S.shape
    if n != m:
        raise ValueError("The matrix S is not square")
    if n % 2 != 0:
        raise ValueError("The matrix S is not of even size")

    nmodes = n // 2

    ch = np.cosh(r) * np.identity(nmodes)
    sh = np.sinh(r) * np.identity(nmodes)
    zh = np.zeros([nmodes, nmodes])
    Schoi = np.block([[ch, sh, zh, zh], [sh, ch, zh, zh], [zh, zh, ch, -sh], [zh, zh, -sh, ch]])

    return expand(S, list(range(nmodes)), n) @ Schoi


def fock_tensor(S, alpha, cutoff, r=np.arcsinh(1.0), check_symplectic=True, rtol=1e-05, atol=1e-08):
    r"""
    Calculates the Fock representation of a Gaussian unitary parametrized by
    the symplectic matrix S and the displacements alpha up to cutoff in Fock space.
    For a complete description of what is being done once the matrix B is obtained
    see:

    * Quesada, N. "Franck-Condon factors by counting perfect matchings of graphs with loops."
    `Journal of Chemical Physics 150, 164113 (2019) <https://aip.scitation.org/doi/10.1063/1.5086387>`_


    Args:
        S (array): symplectic matrix
        alpha (array): complex vector of displacements
        cutoff (int): cutoff in Fock space
        r (float): squeezing parameter used for the Choi expansion
        check_symplectic (boolean): checks whether the input matrix is symplectic
    Return:
        (array): Tensor containing the Fock representation of the Gaussian unitary
    """
    # Check the matrix is symplectic
    if check_symplectic:
        if not is_symplectic(S, rtol=1e-05, atol=1e-08):
            raise ValueError("The matrix S is not symplectic")

    # And that S and alpha have compatible dimensions
    l, _ = S.shape
    if l // 2 != len(alpha):
        raise ValueError("The matrix S and the vector alpha do not have compatible dimensions")

    # Construct its Choi expansion and then the covariance matrix and A matrix of such pure state
    S_exp = choi_expand(S, r)
    cov = S_exp @ S_exp.T
    A = Amat(cov)

    # Because the state is pure then A = B \oplus B^*. We now extract B^* and follow the procedure
    # described in the paper cited above.
    n, _ = A.shape
    N = n // 2
    B = A[0:N, 0:N].conj()

    # Now we need to figure out the loops (cf. Eq. 111 of the reference above)
    l = len(alpha)
    alphat = np.array(list(alpha) + ([0] * l))
    zeta = alphat - B @ alphat.conj()

    # Finally, there are the prefactors (cf. Eq. 113 of the reference above).
    # Note that the factorials that are not included here from Eq. 113 are calculated
    # internally by hafnian_batched when the argument renorm is set to True
    pref_exp = -0.5 * alphat.conj() @ zeta
    R = [1.0 / np.prod((np.tanh(r) ** i) / np.cosh(r)) for i in range(cutoff)]
    # pylint: disable=assignment-from-no-return
    lt = np.arctanh(np.linalg.svd(B, compute_uv=False))
    T = np.exp(pref_exp) / (np.sqrt(np.prod(np.cosh(lt))))

    # Now we use hafnian_batched to get the hafnians of all possible reductions
    tensor = T * hafnian_batched(
        B, cutoff, mu=zeta, renorm=True
    )

    # Finally, we need to renormalize the tensor because we used TMSV to move the input
    # Fock state to the end using auxiliary modes, cf. Fig. 40 and Eq. 60 of the paper
    # cited above.
    vals = list(range(l))
    vals2 = list(range(l, 2 * l))
    tensor_view = tensor.transpose(vals2 + vals)
    # There is probably a better way to do the following rescaling, but this is already "good"
    for p in product(list(range(cutoff)), repeat=l):
        tensor_view[p] = tensor_view[p] * np.prod([R[i] for i in p])
    return tensor
