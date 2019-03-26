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
.. Hafnian Python interface
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
from ._hafnian import (hafnian, hafnian_repeated, haf_int, haf_complex,
                       haf_real, haf_rpt_real, haf_rpt_complex,
                       kron_reduced, permanent_repeated, hafnian_approx, gradhaf)
from ._permanent import perm, perm_real, perm_complex
from ._torontonian import tor, tor_complex, det


__all__ = [
    'hafnian',
    'hafnian_repeated',
    'hafnian_approx',
    'gradhaf',
    'tor',
    'perm',
    'permanent_repeated',
    'det',
    'kron_reduced',
    'version'
]


def version():
    r"""
    Get version number of Hafnian

    Returns:
      str: The package version number
    """
    return __version__
