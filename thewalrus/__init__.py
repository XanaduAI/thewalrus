# Copyright 2021 Xanadu Quantum Technologies Inc.

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
The Walrus
==========

.. currentmodule:: thewalrus

This is the top level module of the The Walrus, containing functions for
computing the hafnian, loop hafnian, and torontonian of matrices.

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

Repeated hafnian algorithm
    The algorithm described in *From moments of sum to moments of product*, :cite:`kan2008moments`.
    This method is more efficient for matrices with repeated rows and columns, and supports calculation of
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

Banded hafnian algorithm
    An algorithm that calculates the hafnian of a matrix :math:`\bm{A}` of size :math:`n \times n` with bandwidth
    :math:`w` by calculating and storing certain subhafnians dictated by the bandwidth. The algorithm
    is described in Section V of
    *Efficient sampling from shallow Gaussian quantum-optical circuits with local interactions*,
    :cite:`qi2020efficient`.

Sparse hafnian algorithm
    An algorithm that calculates the hafnian of a sparse matrix by taking advantage of the Laplace expansion and memoization, to store
    only the relevant paths that contribute non-zero values to the final calculation.

Functions
---------

.. autosummary::
    hafnian
    hafnian_repeated
    hafnian_batched
    tor
    perm
    permanent_repeated
    hermite_multidimensional
    hafnian_banded
    reduction
    version
    low_rank_hafnian

Code details
------------
"""
import thewalrus.quantum
import thewalrus.csamples
import thewalrus.decompositions
import thewalrus.fock_gradients
import thewalrus.labudde
import thewalrus.random
import thewalrus.reference
import thewalrus.samples
import thewalrus.symplectic

from ._hafnian import (
    hafnian,
    loop_hafnian,
    hafnian_repeated,
    reduction,
    hafnian_sparse,
    hafnian_banded,
    matched_reps,
)

from ._low_rank_haf import low_rank_hafnian
from ._hermite_multidimensional import (
    hafnian_batched,
    hermite_multidimensional,
    interferometer,
    grad_hermite_multidimensional,
)
from ._permanent import perm, permanent_repeated

from ._torontonian import (
    tor,
    threshold_detection_prob_displacement,
    threshold_detection_prob,
    numba_tor,
)
from ._version import __version__


__all__ = [
    "hafnian",
    "hafnian_repeated",
    "hafnian_batched",
    "hafnian_sparse",
    "hafnian_banded",
    "loop_hafnian",
    "matched_reps",
    "tor",
    "perm",
    "permanent_repeated",
    "reduction",
    "hermite_multidimensional",
    "grad_hermite_multidimensional",
    "version",
]


def version():
    r"""
    Get version number of The Walrus

    Returns:
      str: The package version number
    """
    return __version__


def about():
    """The Walrus information.

    Prints the installed version numbers for TW and its dependencies,
    and some system info. Please include this information in bug reports.

    **Example:**

    .. code-block:: pycon

        >>> tw.about()
        The Walrus: a Python library for for the calculation of hafnians, Hermite polynomials, and Gaussian boson sampling.
        Copyright 2018-2020 Xanadu Quantum Technologies Inc.

        Python version:            3.8.5
        Platform info:             Linux-5.8.18-1-MANJARO-x86_64-with-arch-Manjaro-Linux
        Installation path:         /home/username/xanadu/thewalrus/thewalrus
        The Walrus version:        0.17.0-dev
        Numpy version:             1.19.5
        Scipy version:             1.7.1
        SymPy version:             1.8
        Numba version:             0.53.1
        Cython version:            0.29.24
    """
    # pylint: disable=import-outside-toplevel
    import os
    import platform

    import sys
    import numpy
    import scipy
    import sympy
    import numba

    # a QuTiP-style infobox
    print(
        "\nThe Walrus: a Python library for for the calculation of hafnians, Hermite polynomials, and Gaussian boson sampling."
    )
    print("Copyright 2018-2021 Xanadu Quantum Technologies Inc.\n")

    print("Python version:            {}.{}.{}".format(*sys.version_info[0:3]))
    print("Platform info:             {}".format(platform.platform()))
    print("Installation path:         {}".format(os.path.dirname(__file__)))
    print("The Walrus version:        {}".format(__version__))
    print("Numpy version:             {}".format(numpy.__version__))
    print("Scipy version:             {}".format(scipy.__version__))
    print("SymPy version:             {}".format(sympy.__version__))
    print("Numba version:             {}".format(numba.__version__))
