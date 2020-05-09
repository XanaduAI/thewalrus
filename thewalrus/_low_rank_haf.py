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
Algorithms for hafnians of low-rank matrices.
"""

from itertools import product
import numpy as np
from sympy import symbols, expand
from scipy.special import factorial2
from repoze.lru import lru_cache


@lru_cache(maxsize=1000000)
def partitions(r, n):
    r"""Returns a list of lists with the r-partitions of the integer :math:`n`, i.e. the r-tuples of non-negative
    integers such that their sum is precisely :math:`n`.

    Note that there are :math:`n + r - 1 \choose r-1` such partitions.

    Args:
        r (int): number of partitions.
        n (int): integer to be partitioned.

    Returns:
        list: r-partitions of n.
    """
    if r == 1:
        return [[n]]

    new_combos = []
    for first_val in range(n + 1):
        rest = partitions(r - 1, n - first_val)
        new = [p[0] + p[1] for p in product([[first_val]], rest)]
        new_combos += new
    return new_combos


def low_rank_hafnian(G):
    r"""Returns the hafnian of the low rank matrix :math:`\bm{A} = \bm{G} \bm{G}^T` where :math:`\bm{G}` is rectangular of size
    :math:`n \times r`  with :math:`r \leq n`.

    Note that the rank of :math:`\bm{A}` is precisely :math:`r`.

    The hafnian is calculated using the algorithm described in Appendix C of
    *A faster hafnian formula for complex matrices and its benchmarking on a supercomputer*,
    :cite:`bjorklund2018faster`.

    Args:
        G (array): factorization of the low rank matrix A = G @ G.T.

    Returns:
        (complex): hafnian of A.
    """
    n, r = G.shape
    if n % 2 != 0:
        return 0
    if r == 1:
        return factorial2(n - 1) * np.prod(G)
    poly = 1
    x = symbols("x0:" + str(r))
    for k in range(n):
        term = 0
        for j in range(r):
            term += G[k, j] * x[j]
        poly = expand(poly * term)

    comb = partitions(r, n // 2)
    haf_val = 0.0
    for c in comb:
        monomial = 1
        facts = 1
        for i, pi in enumerate(c):
            monomial *= x[i] ** (2 * pi)
            facts = facts * factorial2(2 * pi - 1)
        haf_val += complex(poly.coeff(monomial) * facts)
    return haf_val
