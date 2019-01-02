# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Hafnian Python interface
========================

.. currentmodule:: hafnian

This is the top level module of the Hafnian Python interface,
containing the function :func:`hafnian` and :func:`perm`.
These wrapper functions determine,
based on the input matrix, whether to use the complex or real
C++/Fortran library.


Python wrappers
---------------

.. autosummary::
   hafnian
   perm
   version

For more advanced usage, access to the libraries directly are provided
via the functions:

* :func:`haf_real` links to ``hafnian.lib.libhaf.haf_real``
* :func:`haf_complex` links to ``hafnian.lib.libhaf.haf_complex``
* :func:`haf_int` links to ``hafnian.lib.libhaf.haf_int``
* :func:`perm_real` links to ``hafnian.lib.libperm.re``.
* :func:`perm_complex` links to ``hafnian.lib.libperm.comp``.


Code details
------------
"""
#pylint: disable=wrong-import-position
import os
import platform

import numpy as np

if platform.system() == 'Windows': # pragma: no cover
    extra_dll_dir = os.path.join(os.path.dirname(__file__), '.libs')
    if os.path.isdir(extra_dll_dir):
        os.environ["PATH"] += os.pathsep + extra_dll_dir

from ._version import __version__
from ._hafnian import hafnian, haf_int, haf_complex, haf_real
from ._permanent import perm, perm_real, perm_complex


__all__ = [
    'hafnian',
    'perm',
    'version'
]


def version():
    r"""
    Get version number of Hafnian

    Returns:
      str: The package version number
    """
    return __version__
