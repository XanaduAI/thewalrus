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
"""Tests for the Python reference hafnian functions"""
# pylint: disable=no-self-use,redefined-outer-name
import pytest

import numpy as np
from scipy.special import factorial2

from thewalrus.reference import T, spm, pmp, hafnian


class TestReferenceHafnian:
    """Tests for the reference hafnian"""

    def test_telephone(self):
        r""" Checks that the function T produces the first 11 telephone numbers"""
        tn = np.array([1, 1, 2, 4, 10, 26, 76, 232, 764, 2620, 9496])
        Tn = np.array([T(n) for n in range(len(tn))])
        assert np.all(tn == Tn)

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9, 10])
    def test_spm(self, n):
        r"""Checks that the number of elements in spm(n) is precisely the n^th telephone number"""
        length = len(list(spm(tuple(range(n)))))
        assert np.allclose(length, T(n))

    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_pmp(self, n):
        r"""Checks that the number of elements in pmp(n) is precisely the (n-1)!! for even n"""
        length = len(list(pmp(tuple(range(n)))))
        assert np.allclose(length, factorial2(n - 1))

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6])
    def test_hafnian(self, n):
        r"""Checks that the hafnian of the all ones matrix of size n is (n-1)!!"""
        M = np.ones([n, n])
        if n % 2 == 0:
            assert np.allclose(factorial2(n - 1), hafnian(M))
        else:
            assert np.allclose(0, hafnian(M))

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6])
    def test_loophafnian(self, n):
        r"""Checks that the loop hafnian of the all ones matrix of size n is T(n)"""
        M = np.ones([n, n])
        assert np.allclose(T(n), hafnian(M, loop=True))
