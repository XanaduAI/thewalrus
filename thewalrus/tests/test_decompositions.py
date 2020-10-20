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
"""Decomposition tests"""
# pylint: disable=no-self-use, assignment-from-no-return
import numpy as np
import pytest
from scipy.linalg import block_diag

from thewalrus.random import random_interferometer as haar_measure
from thewalrus.decompositions import williamson
from thewalrus.symplectic import sympmat as omega

class TestWilliamsonDecomposition:
    """Tests for the Williamson decomposition"""

    @pytest.fixture
    def create_cov(self, hbar, tol):
        """create a covariance state for use in testing.

        Args:
            nbar (array[float]): vector containing thermal state values

        Returns:
            tuple: covariance matrix and symplectic transform
        """

        def _create_cov(nbar):
            """wrapped function"""
            n = len(nbar)
            O = omega(n)

            # initial vacuum state
            cov = np.diag(2 * np.tile(nbar, 2) + 1) * hbar / 2

            # interferometer 1
            U1 = haar_measure(n)
            S1 = np.vstack([np.hstack([U1.real, -U1.imag]), np.hstack([U1.imag, U1.real])])

            # squeezing
            r = np.log(0.2 * np.arange(n) + 2)
            Sq = block_diag(np.diag(np.exp(-r)), np.diag(np.exp(r)))

            # interferometer 2
            U2 = haar_measure(n)
            S2 = np.vstack([np.hstack([U2.real, -U2.imag]), np.hstack([U2.imag, U2.real])])

            # final symplectic
            S_final = S2 @ Sq @ S1

            # final covariance matrix
            cov_final = S_final @ cov @ S_final.T

            # check valid symplectic transform
            assert np.allclose(S_final.T @ O @ S_final, O)

            # check valid state
            eigs = np.linalg.eigvalsh(cov_final + 1j * (hbar / 2) * O)
            eigs[np.abs(eigs) < tol] = 0
            assert np.all(eigs >= 0)

            if np.allclose(nbar, 0):
                # check pure
                assert np.allclose(np.linalg.det(cov_final), (hbar / 2) ** (2 * n))
            else:
                # check not pure
                assert not np.allclose(np.linalg.det(cov_final), (hbar / 2) ** (2 * n))

            return cov_final, S_final

        return _create_cov

    def test_square_validation(self):
        """Test that the graph_embed decomposition raises exception if not square"""
        A = np.random.rand(4, 5) + 1j * np.random.rand(4, 5)
        with pytest.raises(ValueError, match="matrix is not square"):
            williamson(A)

    def test_symmetric_validation(self):
        """Test that the graph_embed decomposition raises exception if not symmetric"""
        A = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
        with pytest.raises(ValueError, match="matrix is not symmetric"):
            williamson(A)

    def test_even_validation(self):
        """Test that the graph_embed decomposition raises exception if not even number of rows"""
        A = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
        A += A.T
        with pytest.raises(ValueError, match="must have an even number of rows/columns"):
            williamson(A)

    def test_positive_definite_validation(self):
        """Test that the graph_embed decomposition raises exception if not positive definite"""
        A = np.diag([-2, 0.1, 2, 3])
        with pytest.raises(ValueError, match="matrix is not positive definite"):
            williamson(A)

    def test_vacuum_state(self, tol):
        """Test vacuum state"""
        V = np.identity(4)
        Db, S = williamson(V)
        assert np.allclose(Db, np.identity(4), atol=tol, rtol=0)
        assert np.allclose(S, np.identity(4), atol=tol, rtol=0)

    def test_pure_state(self, create_cov, hbar, tol):
        """Test pure state"""
        n = 3
        O = omega(n)

        cov, _ = create_cov(np.zeros([n]))

        Db, S = williamson(cov)
        nbar = np.diag(Db) / hbar - 0.5

        # check decomposition is correct
        assert np.allclose(S @ Db @ S.T, cov, atol=tol, rtol=0)
        # check nbar = 0
        assert np.allclose(nbar, 0, atol=tol, rtol=0)
        # check S is symplectic
        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)

    def test_mixed_state(self, create_cov, hbar, tol):
        """Test mixed state"""
        n = 3
        O = omega(n)
        nbar_in = np.abs(np.random.rand(n))

        cov, _ = create_cov(nbar_in)

        Db, S = williamson(cov)
        nbar = np.diag(Db) / hbar - 0.5

        # check decomposition is correct
        assert np.allclose(S @ Db @ S.T, cov, atol=tol, rtol=0)
        # check nbar
        assert np.allclose(sorted(nbar[:n]), sorted(nbar_in), atol=tol, rtol=0)
        # check S is symplectic
        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)
