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
"""
Unit tests for the :mod:`thewalrus` configuration class :class:`Configuration`.
"""
# pylint: disable=protected-access

import contextlib
import io
import re

import pytest

import thewalrus as tw


def test_about():
    """
    about: Tests if the about string prints correctly.
    """
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        tw.about()
    out = f.getvalue().strip()

    assert "Python version:" in out
    pl_version_match = re.search(r"The Walrus version:\s+([\S]+)\n", out).group(1)
    assert tw.version() in pl_version_match
    assert "Numpy version" in out
    assert "Scipy version" in out
    assert "SymPy version" in out
    assert "Numba version" in out
