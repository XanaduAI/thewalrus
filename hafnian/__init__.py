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

.. currentmodule:: hafnian

This is the top level module of the Hafnian Python interface,
containing functions for computing the hafnian, loop hafnian,
and torontonian of matrices.

Algorithm terminology
---------------------

Eigenvalue hafnian algorithm
    The algorithm described in
    *A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer*,
    `arxiv:1805.12498 <https://arxiv.org/abs/1805.12498>`__.
    This algorithm scales like :math:`\mathcal{O}(n^3 2^{n/2})`, and supports caclulation of
    the loop hafnian.

Recursive hafnian algorithm
    The algorithm described in *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`.
    This algorithm scales like :math:`\mathcal{O}(n^4 2^{n/2})`. This algorithm does not
    currently support the loop hafnian.

Repeating hafnian algorithm
    The algorithm described in *From moments of sum to moments of product*,
    `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__.
    This method is more efficient for matrices with repeated rows and columns, and supports caclulation of
    the loop hafnian.

Approximate hafnian algorithm
    An algorithm that allows us to efficiently approximate the hafnian of
    matrices with non-negative elements. This is done by sampling determinants;
    the larger the number of samples taken, the higher the accuracy.


Python wrappers
---------------

.. autosummary::
    hafnian
    hafnian_repeated
    tor
    perm
    permanent_repeated
    reduction
    version
"""
# pylint: disable=wrong-import-position
import os
import platform

import numpy as np

if platform.system() == "Windows":  # pragma: no cover
    extra_dll_dir = os.path.join(os.path.dirname(__file__), ".libs")
    if os.path.isdir(extra_dll_dir):
        os.environ["PATH"] += os.pathsep + extra_dll_dir

from ._version import __version__
from ._hafnian import (
    hafnian,
    hafnian_repeated,
    haf_int,
    haf_complex,
    haf_real,
    haf_rpt_real,
    haf_rpt_complex,
    reduction,
)

from ._torontonian import tor

from ._permanent import perm, perm_real, perm_complex, permanent_repeated

__all__ = [
    "hafnian",
    "hafnian_repeated",
    "tor",
    "perm",
    "permanent_repeated",
    "reduction",
    "version",
]


def version():
    r"""
    Get version number of Hafnian

    Returns:
      str: The package version number
    """
    return __version__
