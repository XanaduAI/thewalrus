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
"""Tests for the rlhaf Python function, which calls rlhafnian.so"""

import unittest

import numpy as np
from math import factorial as fac
from hafnian.rlhaf import hafnian


class TestRlhaf(unittest.TestCase):
    """Various Hafnian consistency checks"""

    def setUp(self):
        """Set up"""
        self.n = 6

    def test_2x2(self):
        """Check 2x2 hafnian"""
        A = np.float64(np.random.random([2, 2]))
        A = A + A.T
        haf = hafnian(A)
        self.assertTrue(np.allclose(haf, A[0, 1]))

    def test_4x4(self):
        """Check 4x4 hafnian"""
        A = np.float64(np.random.random([4, 4]))
        A += A.T
        haf = hafnian(A)
        expected = A[0, 1]*A[2, 3] + \
            A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
        self.assertTrue(np.allclose(haf, expected))


    def test_identity(self):
        """Check hafnian(I)=0"""
        A = np.identity(self.n)
        haf = hafnian(A)
        self.assertEqual(haf, 0)

    def test_ones(self):
        """Check hafnian(J_2n)=(2n)!/(n!2^n)"""
        A = np.float64(np.ones([2*self.n, 2*self.n]))
        haf = hafnian(A)
        expected = fac(2*self.n)/(fac(self.n)*(2**self.n))
        self.assertTrue(np.allclose(haf, expected))

    def test_integer_casting(self):
        """Check casting to integer"""
        A = np.int64(np.ones([2*self.n, 2*self.n]))
        haf = hafnian(A)
        expected = fac(2*self.n)/(fac(self.n)*(2**self.n))
        self.assertTrue(np.allclose(haf, expected))

    def test_block_ones(self):
        """Check hafnian([[0, I_n], [I_n, 0]])=n!"""
        O = np.zeros([self.n, self.n])
        B = np.ones([self.n, self.n])
        A = np.vstack([np.hstack([O, B]),
                       np.hstack([B, O])])
        A = np.float64(A)
        haf = hafnian(A)
        expected = float(fac(self.n))
        self.assertTrue(np.allclose(haf, expected))


if __name__ == '__main__':
    unittest.main()
