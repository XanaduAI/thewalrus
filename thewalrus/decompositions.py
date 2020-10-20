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
This module implements common shared matrix decompositions that are
used to perform gate decompositions.
"""
import numpy as np

from scipy.linalg import block_diag, sqrtm, schur
from thewalrus.symplectic import sympmat

def williamson(V, rtol=1e-05, atol=1e-08):
    r"""Williamson decomposition of positive-definite (real) symmetric matrix.

    See https://math.stackexchange.com/questions/1171842/finding-the-symplectic-matrix-in-williamsons-theorem/2682630#2682630
	and https://strawberryfields.ai/photonics/conventions/decompositions.html#williamson-decomposition
    Args:
        V (array[float]): positive definite symmetric (real) matrix
        rtol (float): the relative tolerance parameter used in ``np.allclose``
        atol (float): the absolute tolerance parameter used in ``np.allclose``

    Returns:
        tuple[array,array]: ``(Db, S)`` where ``Db`` is a diagonal matrix
            and ``S`` is a symplectic matrix such that :math:`V = S^T Db S`
    """
    (n, m) = V.shape

    if n != m:
        raise ValueError("The input matrix is not square")

    if not np.allclose(V, V.T, rtol=rtol, atol=atol):
        raise ValueError("The input matrix is not symmetric")

    if n % 2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n // 2
    omega = sympmat(n)
    vals = np.linalg.eigvalsh(V)

    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    Mm12 = sqrtm(np.linalg.inv(V)).real
    r1 = Mm12 @ omega @ Mm12
    s1, K = schur(r1)
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    seq = []

    # In what follows I construct a permutation matrix p  so that the Schur matrix has
    # only positive elements above the diagonal
    # Also the Schur matrix uses the x_1,p_1, ..., x_n,p_n  ordering thus I permute using perm
    # to go to the ordering x_1, ..., x_n, p_1, ... , p_n

    for i in range(n):
        if s1[2 * i, 2 * i + 1] > 0:
            seq.append(I)
        else:
            seq.append(X)
    perm = np.array([2*i for i in range(n)] + [2*i+1 for i in range(n)])
    p = block_diag(*seq)
    Kt = K @ p
    Ktt = Kt[:,perm]
    s1t = p @ s1 @ p
    dd = [1/s1t[2 * i, 2 * i + 1] for i in range(n)]
    Db = np.diag(dd+dd)
    S = Mm12 @ Ktt @ sqrtm(Db)
    return Db, np.linalg.inv(S).T
