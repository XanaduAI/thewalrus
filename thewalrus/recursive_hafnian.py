# Copyright 2022 Xanadu Quantum Technologies Inc.

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
This module computes the hafnian of a matrix with a recursive algorithm
as described in described in *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`.
The code was adapted from c++ with the code found on:
https://github.com/XanaduAI/thewalrus/blob/v0.17.0/include/recursive_hafnian.hpp
"""
# pylint: disable=too-many-branches
import numpy as np
import numba


@numba.jit
def hafnian(m):
    r"""Computes the hafnian of the matrix with the recursive algorithm. It is an implementation of
    algorithm 2 in *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`.
    Args:
        m (array): the input matrix
    Returns:
        float: the hafnian of the input matrix
    """
    nb_lines = len(m)
    nb_columns = len(m[0])
    if nb_lines != nb_columns:
        raise ValueError("Matrix must be square")

    if nb_lines % 2 != 0:
        raise ValueError("Matrix size must be even")

    n = int(float(len(m)) / 2)
    z = np.zeros((n * (2 * n - 1), n + 1))
    for j in range(1, 2 * n):
        for k in range(j):
            z[int(j * (j - 1) / 2 + k)][0] = m.copy()[j][k]
    g = np.zeros(n + 1)
    g[0] = 1
    return solve(z, 2 * n, 1, g, n)


@numba.jit
def solve(b, s, w, g, n):
    r"""Implements the recursive algorithm.
    Args:
        b (array): matrix that is transformed recursively
        s (int): size of the original matrix that changes at every recursion
        k (int): a variable of the recursive algorithm
        g (int): matrix that is transformed recursively
        n (int): size of the original matrix divided by 2
    Returns:
        float: the hafnian of the input matrix
    """
    if s == 0:
        return w * g[n]
    c = np.zeros((int((s - 2) * (s - 3) / 2), n + 1))
    i = 0
    for j in range(1, s - 2):
        for k in range(j):
            c[i] = b[int((j + 1) * (j + 2) / 2 + k + 2)]
            i += 1
    h = solve(c, s - 2, -w, g, n)
    e = g[:].copy()
    for u in range(n):
        for v in range(n - u):
            e[u + v + 1] += g[u] * b[0][v]
    for j in range(1, s - 2):
        for k in range(j):
            for u in range(n):
                for v in range(n - u):
                    c[int(j * (j - 1) / 2 + k)][u + v + 1] += (
                        b[int((j + 1) * (j + 2) / 2)][u]
                        * b[int((k + 1) * (k + 2) / 2 + 1)][v]
                        + b[int((k + 1) * (k + 2) / 2)][u]
                        * b[int((j + 1) * (j + 2) / 2 + 1)][v]
                    )
    return h + solve(c, s - 2, w, e, n)
