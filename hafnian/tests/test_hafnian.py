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
"""Tests for the Python hafnian wrapper function"""
import unittest

import numpy as np
import hafnian as hf
from hafnian import hafnian
from hafnian.lib.libhaf import haf_complex, haf_real, haf_int


class TestVersion(unittest.TestCase):
    """Test the version number is correctly imported"""
    def test_version_number(self):
        """returns true if returns a string"""
        res = hf.version()
        self.assertTrue(isinstance(res, str))


class TestPythonInterfaceWrapper(unittest.TestCase):
    """Various Hafnian consistency checks"""

    def test_array_exception(self):
        """Check exception for non-matrix argument"""
        with self.assertRaises(TypeError):
            hafnian(1)

    def test_square_exception(self):
        """Check exception for non-square argument"""
        A = np.zeros([2, 3])
        with self.assertRaises(ValueError):
            hafnian(A)

    def test_odd_dim(self):
        """Check hafnian for matrix with odd dimensions"""
        A = np.zeros([3, 3])
        self.assertEqual(hafnian(A), 0)

    def test_non_symmetric_exception(self):
        """Check exception for non-symmetric matrix"""
        A = np.ones([4, 4])
        A[0, 1] = 0.
        with self.assertRaises(ValueError):
            hafnian(A)

    def test_nan(self):
        """Check exception for non-finite matrix"""
        A = np.array([[2, 1], [1, np.nan]])
        with self.assertRaises(ValueError):
            hafnian(A)

    def test_2x2(self):
        """Check 2x2 hafnian"""
        A = np.random.random([2, 2])
        A += A.T
        haf = hafnian(A)
        self.assertEqual(haf, A[0, 1])
        haf = hafnian(A, loop=True)
        self.assertEqual(haf, A[0, 1]+A[0, 0]*A[1, 1])

    def test_3x3(self):
        """Check 3x3 hafnian"""
        A = np.ones([3, 3])
        haf = hafnian(A)
        self.assertEqual(haf, 0.0)
        haf = hafnian(A, loop=True)
        self.assertEqual(haf, 4.0)

    def test_4x4(self):
        """Check 4x4 hafnian"""
        A = np.random.random([4, 4])
        A += A.T
        haf = hafnian(A)
        expected = A[0, 1]*A[2, 3] + \
            A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
        self.assertEqual(haf, expected)

        haf = hafnian(A, loop=True)
        expected = A[0, 1]*A[2, 3] \
            + A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2] \
            + A[0, 0]*A[1, 1]*A[2, 3] + A[0, 1]*A[2, 2]*A[3, 3] \
            + A[0, 2]*A[1, 1]*A[3, 3] + A[0, 0]*A[2, 2]*A[1, 3] \
            + A[0, 0]*A[3, 3]*A[1, 2] + A[0, 3]*A[1, 1]*A[2, 2] \
            + A[0, 0]*A[1, 1]*A[2, 2]*A[3, 3]
        self.assertEqual(haf, expected)

    def test_real(self):
        """Check hafnian(A)=haf_real(A) for a random
        real matrix.
        """
        A = np.random.random([6, 6])
        A += A.T
        haf = hafnian(A)
        expected = haf_real(A)
        self.assertTrue(np.allclose(haf, expected))

        haf = hafnian(A, loop=True)
        expected = haf_real(A, loop=True)
        self.assertTrue(np.allclose(haf, expected))

        A = np.random.random([6, 6])
        A += A.T
        A = np.array(A, dtype=np.complex128)
        haf = hafnian(A)
        expected = haf_real(np.float64(A.real))
        self.assertTrue(np.allclose(haf, expected))

    def test_complex(self):
        """Check hafnian(A)=haf_complex(A) for a random
        real matrix.
        """
        A = np.complex128(np.random.random([6, 6]))
        A += 1j*np.random.random([6, 6])
        A += A.T
        haf = hafnian(A)
        expected = haf_complex(A)
        self.assertTrue(np.allclose(haf, expected))

        haf = hafnian(A, loop=True)
        expected = haf_complex(A, loop=True)
        self.assertTrue(np.allclose(haf, expected))

    def test_int(self):
        """Check hafnian(A)=haf_int(A) for a random
        real matrix.
        """
        A = np.ones([6, 6])
        haf = hafnian(A)
        expected = haf_int(np.int64(A))
        self.assertTrue(np.allclose(haf, expected))

if __name__ == '__main__':
    unittest.main()
