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
r"""
Python library
==============

.. currentmodule:: thewalrus

This is the top level module of the The Walrus Python interface,
containing functions for computing the hafnian, loop hafnian,
and torontonian of matrices.

Algorithm terminology
---------------------

Eigenvalue hafnian algorithm
    The algorithm described in
    *A faster hafnian formula for complex matrices and its benchmarking on a supercomputer*,
    :cite:`bjorklund2018faster`.
    This algorithm scales like :math:`\mathcal{O}(n^3 2^{n/2})`, and supports calculation of
    the loop hafnian.

Recursive hafnian algorithm
    The algorithm described in *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`.
    This algorithm scales like :math:`\mathcal{O}(n^4 2^{n/2})`. This algorithm does not
    currently support the loop hafnian.

Repeating hafnian algorithm
    The algorithm described in *From moments of sum to moments of product*, :cite:`kan2008moments`.
    This method is more efficient for matrices with repeated rows and columns, and supports caclulation of
    the loop hafnian.

Approximate hafnian algorithm
    The algorithm described in *Polynomial time algorithms to approximate permanents and mixed discriminants
    within a simply exponential factor*, :cite:`barvinok1999polynomial`.
    This algorithm allows us to efficiently approximate the hafnian of
    matrices with non-negative elements. This is done by sampling determinants;
    the larger the number of samples taken, the higher the accuracy.

Batched hafnian algorithm
    An algorithm that allows the calculation of hafnians of all reductions of
    a given matrix up to the cutoff (resolution) provided. Internally, this algorithm
    makes use of the multidimensional Hermite polynomials as per
    *The calculation of multidimensional Hermite polynomials and Gram-Charlier coefficients*
    :cite:`berkowitz1970calculation`.

Low-rank hafnian algorithm
    An algorithm that allows to calculate the hafnian of an :math:`r`-rank matrix :math:`\bm{A}` of size :math:`n \times n`
    by factorizing it as :math:`\bm{A} = \bm{G} \bm{G}^T` where :math:`\bm{G}` is of size :math:`n \times r`. The algorithm
    is described in Appendix C of
    *A faster hafnian formula for complex matrices and its benchmarking on a supercomputer*,
    :cite:`bjorklund2018faster`.


Python wrappers
---------------

.. autosummary::
    hafnian
    hafnian_repeated
    hafnian_batched
    tor
    perm
    permanent_repeated
    hermite_multidimensional

Pure Python functions
---------------------

.. autosummary::
    reduction
    version
    low_rank_hafnian
"""
# pylint: disable=wrong-import-position
import os
import platform

import numpy as np

import thewalrus.quantum

from ._hafnian import (
    haf_complex,
    haf_int,
    haf_real,
    haf_rpt_complex,
    haf_rpt_real,
    hafnian,
    hafnian_repeated,
    reduction,
)
from ._low_rank_haf import low_rank_hafnian
from ._hermite_multidimensional import hafnian_batched, hermite_multidimensional
from ._permanent import perm, perm_complex, perm_real, permanent_repeated
from ._torontonian import tor
from ._version import __version__


__all__ = [
    "hafnian",
    "hafnian_repeated",
    "hafnian_batched",
    "tor",
    "perm",
    "permanent_repeated",
    "reduction",
    "hermite_multidimensional",
    "version",
]


def version():
    r"""
    Get version number of The Walrus

    Returns:
      str: The package version number
    """
    return __version__
