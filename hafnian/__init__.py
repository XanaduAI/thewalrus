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
"""FPMW"""

import numpy as np

from ._version import __version__
from ._hafnian import hafnian, haf_complex, haf_real

__all__ = ['version']


def version():
    r"""
    Get version number of Hafnian

    Returns:
      str: The package version number
    """
    return __version__
