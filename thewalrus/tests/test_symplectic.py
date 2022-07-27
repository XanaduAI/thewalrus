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
"""Symplectic tests"""
# pylint: disable=no-self-use, assignment-from-no-return
import pytest

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import (
    csc_array,
    csr_array,
    bsr_array,
    lil_array,
    dok_array,
    coo_array,
    dia_array,
    issparse,
)

from thewalrus import symplectic
from thewalrus.quantum import is_valid_cov
from thewalrus.random import random_symplectic


# pylint: disable=too-few-public-methods
class TestVacuum:
    """Tests for the vacuum_state function"""

    def test_vacuum_state(self, hbar, tol):
        """Test the vacuum state is correct."""
        modes = 3
        means, cov = symplectic.vacuum_state(modes, hbar=hbar)
        assert np.allclose(means, np.zeros([2 * modes]), atol=tol, rtol=0)
        assert np.allclose(cov, np.identity(2 * modes) * hbar / 2, atol=tol, rtol=0)


class TestSqueezing:
    """Tests for the squeezing symplectic"""

    def test_symplectic(self, tol):
        """Test that the squeeze operator is symplectic"""
        r = 0.543
        phi = 0.123
        S = symplectic.squeezing(r, phi)

        # the symplectic matrix
        O = np.array([[0, 1], [-1, 0]])

        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)

    def test_squeezing(self, tol):
        """Test the squeezing symplectic transform."""
        r = 0.543
        phi = 0.123
        S = symplectic.squeezing(r, phi)
        out = S @ S.T

        # apply to an identity covariance matrix
        rotation = np.array(
            [[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]]
        )
        expected = rotation @ np.diag(np.exp([-2 * r, 2 * r])) @ rotation.T
        assert np.allclose(out, expected, atol=tol, rtol=0)

    def test_squeezing_no_phi(self, tol):
        """Test the squeezing symplectic transform without specifying phi"""
        r = 0.543
        phi = 0.0
        S = symplectic.squeezing(r)
        out = S @ S.T

        # apply to an identity covariance matrix
        rotation = np.array(
            [[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]]
        )
        expected = rotation @ np.diag(np.exp([-2 * r, 2 * r])) @ rotation.T
        assert np.allclose(out, expected, atol=tol, rtol=0)

    def test_squeezing_no_phi_array(self, tol):
        """Test multimode squeezing symplectic transform without specifying phi"""
        r = np.random.randn(6)
        phi = np.zeros_like(r)

        S = symplectic.squeezing(r)
        S_phi = symplectic.squeezing(r, phi)

        assert np.allclose(S, S_phi, atol=tol, rtol=0)

    def test_symplectic_multimode(self, tol):
        """Test multimode version gives symplectic matrix"""
        r = [0.543] * 4
        phi = [0.123] * 4
        S = symplectic.squeezing(r, phi)

        # the symplectic matrix
        O = symplectic.sympmat(4)

        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)

    def test_dtype(self, tol):
        """Test multimode version gives symplectic matrix"""
        r = [0.543] * 4
        phi = [0.123] * 4
        S = symplectic.squeezing(r, phi)

        S32_bit = symplectic.squeezing(r, phi, dtype=np.float32)

        # the symplectic matrix
        O = symplectic.sympmat(4)

        assert np.allclose(S32_bit @ O @ S32_bit.T, O, atol=tol, rtol=0)
        assert np.allclose(S, S32_bit, atol=tol, rtol=0)


class TestTwoModeSqueezing:
    """Tests for the TMS symplectic"""

    def test_symplectic(self, tol):
        """Test that the two mode squeeze operator is symplectic"""
        r = 0.543
        phi = 0.123
        S = symplectic.expand(symplectic.two_mode_squeezing(r, phi), modes=[0, 2], N=4)

        # the symplectic matrix
        O = np.block([[np.zeros([4, 4]), np.identity(4)], [-np.identity(4), np.zeros([4, 4])]])

        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)

    def test_decompose(self, tol):
        """Test the two mode squeezing symplectic transform decomposes correctly."""
        r = 0.543
        phi = 0.123
        S = symplectic.two_mode_squeezing(r, phi)

        # test that S = B^\dagger(pi/4, 0) [S(z) x S(-z)] B(pi/4)
        # fmt:off
        B = np.array([[1, -1, 0, 0], [1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]])/np.sqrt(2)

        Sq1 = np.array([[np.cosh(r)-np.cos(phi)*np.sinh(r), -np.sin(phi)*np.sinh(r)],
                        [-np.sin(phi)*np.sinh(r), np.cosh(r)+np.cos(phi)*np.sinh(r)]])

        Sq2 = np.array([[np.cosh(-r)-np.cos(phi)*np.sinh(-r), -np.sin(phi)*np.sinh(-r)],
                        [-np.sin(phi)*np.sinh(-r), np.cosh(-r)+np.cos(phi)*np.sinh(-r)]])
        # fmt:on

        Sz = block_diag(Sq1, Sq2)[:, [0, 2, 1, 3]][[0, 2, 1, 3]]
        expected = B.conj().T @ Sz @ B
        assert np.allclose(S, expected, atol=tol, rtol=0)

    def test_coherent(self, hbar, tol):
        """Test the two mode squeezing symplectic transform acts correctly
        on coherent states"""
        r = 0.543
        phi = 0.123
        S = symplectic.two_mode_squeezing(r, phi)

        # test that S |a1, a2> = |ta1+ra2, ta2+ra1>
        a1 = 0.23 + 0.12j
        a2 = 0.23 + 0.12j
        out = S @ np.array([a1.real, a2.real, a1.imag, a2.imag]) * np.sqrt(2 * hbar)

        T = np.cosh(r)
        R = np.exp(1j * phi) * np.sinh(r)
        a1out = T * a1 + R * np.conj(a2)
        a2out = T * a2 + R * np.conj(a1)
        expected = np.array([a1out.real, a2out.real, a1out.imag, a2out.imag]) * np.sqrt(2 * hbar)
        assert np.allclose(out, expected, atol=tol, rtol=0)


class TestInterferometer:
    """Tests for the interferometer"""

    def test_interferometer(self, tol):
        """Test that an interferometer returns correct symplectic"""
        # fmt:off
        U = np.array([[0.83645892-0.40533293j, -0.20215326+0.30850569j],
                      [-0.23889780-0.28101519j, -0.88031770-0.29832709j]])
        # fmt:on

        S = symplectic.interferometer(U)
        expected = np.block([[U.real, -U.imag], [U.imag, U.real]])

        assert np.allclose(S, expected, atol=tol, rtol=0)

    def test_symplectic(self, tol):
        """Test that the interferometer is symplectic"""
        # random interferometer
        # fmt:off
        U = np.array([[-0.06658906-0.36413058j, 0.07229868+0.65935896j, 0.59094625-0.17369183j, -0.18254686-0.10140904j],
                      [0.53854866+0.36529723j, 0.61152793+0.15022026j, 0.05073631+0.32624882j, -0.17482023-0.20103772j],
                      [0.34818923+0.51864844j, -0.24334624+0.0233729j, 0.3625974 -0.4034224j, 0.10989667+0.49366039j],
                      [0.16548085+0.14792642j, -0.3012549 -0.11387682j, -0.12731847-0.44851389j, -0.55816075-0.5639976j]])
        # fmt:on
        U = symplectic.interferometer(U)

        # the symplectic matrix
        O = np.block([[np.zeros([4, 4]), np.identity(4)], [-np.identity(4), np.zeros([4, 4])]])

        assert np.allclose(U @ O @ U.T, O, atol=tol, rtol=0)

    def test_50_50_beamsplitter(self, tol):
        """Test that an interferometer returns correct symplectic for a 50-50 beamsplitter"""
        U = np.array([[1, -1], [1, 1]]) / np.sqrt(2)

        S = symplectic.interferometer(U)
        B = np.array([[1, -1, 0, 0], [1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]]) / np.sqrt(2)

        assert np.allclose(S, B, atol=tol, rtol=0)

    def test_beamsplitter(self, tol):
        """Test that an interferometer returns correct symplectic for an arbitrary beamsplitter"""
        theta = 0.98
        phi = 0.41
        U = symplectic.beam_splitter(theta, phi)
        S = symplectic.interferometer(U)
        expected = np.block([[U.real, -U.imag], [U.imag, U.real]])
        np.allclose(S, expected, atol=tol, rtol=0)

    def test_rotation(self, tol):
        """Test that a rotation returns the correct symplectic for an abritrary angle"""
        theta = 0.98
        S = symplectic.rotation(theta)
        expected = np.block([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        np.allclose(S, expected, atol=tol, rtol=0)


class TestPassiveTransformation:
    """tests for linear transformation"""

    def test_transformation(self, tol):
        """Test that an transformation returns the correct state"""

        M = 4
        cov = np.arange(4 * M**2, dtype=np.float64).reshape((2 * M, 2 * M))
        mu = np.arange(2 * M, dtype=np.float64)

        T = np.sqrt(0.9) * M ** (-0.5) * np.ones((6, M), dtype=np.float64)

        mu_out, cov_out = symplectic.passive_transformation(mu, cov, T)
        # fmt:off
        expected_mu = np.array([ 2.84604989,  2.84604989,  2.84604989,  2.84604989,  2.84604989,
                                 2.84604989, 10.43551628, 10.43551628, 10.43551628, 10.43551628,
                                 10.43551628, 10.43551628])
        expected_cov = np.array([
            [ 48.7,  47.7,  47.7,  47.7,  47.7,  47.7,  63. ,  63. ,  63. , 63. ,  63. ,  63. ],
            [ 47.7,  48.7,  47.7,  47.7,  47.7,  47.7,  63. ,  63. ,  63. , 63. ,  63. ,  63. ],
            [ 47.7,  47.7,  48.7,  47.7,  47.7,  47.7,  63. ,  63. ,  63. , 63. ,  63. ,  63. ],
            [ 47.7,  47.7,  47.7,  48.7,  47.7,  47.7,  63. ,  63. ,  63. , 63. ,  63. ,  63. ],
            [ 47.7,  47.7,  47.7,  47.7,  48.7,  47.7,  63. ,  63. ,  63. , 63. ,  63. ,  63. ],
            [ 47.7,  47.7,  47.7,  47.7,  47.7,  48.7,  63. ,  63. ,  63. , 63. ,  63. ,  63. ],
            [163.8, 163.8, 163.8, 163.8, 163.8, 163.8, 178.3, 177.3, 177.3, 177.3, 177.3, 177.3],
            [163.8, 163.8, 163.8, 163.8, 163.8, 163.8, 177.3, 178.3, 177.3, 177.3, 177.3, 177.3],
            [163.8, 163.8, 163.8, 163.8, 163.8, 163.8, 177.3, 177.3, 178.3, 177.3, 177.3, 177.3],
            [163.8, 163.8, 163.8, 163.8, 163.8, 163.8, 177.3, 177.3, 177.3, 178.3, 177.3, 177.3],
            [163.8, 163.8, 163.8, 163.8, 163.8, 163.8, 177.3, 177.3, 177.3, 177.3, 178.3, 177.3],
            [163.8, 163.8, 163.8, 163.8, 163.8, 163.8, 177.3, 177.3, 177.3, 177.3, 177.3, 178.3]])
        # fmt:on

        assert np.allclose(mu_out, expected_mu, atol=tol, rtol=0)
        assert np.allclose(cov_out, expected_cov, atol=tol, rtol=0)

    @pytest.mark.parametrize("M", range(1, 10))
    def test_valid_cov(self, M, tol):
        """test that the output is a valid covariance matrix, even when not square"""
        a = np.arange(4 * M**2, dtype=np.float64).reshape((2 * M, 2 * M))
        cov = a @ a.T + np.eye(2 * M)
        mu = np.arange(2 * M, dtype=np.float64)

        T = np.sqrt(0.9) * M ** (-0.5) * np.ones((6, M), dtype=np.float64)

        mu_out, cov_out = symplectic.passive_transformation(mu, cov, T)

        assert cov_out.shape == (12, 12)
        assert len(mu_out) == 12
        assert is_valid_cov(cov_out, atol=tol, rtol=0)

    @pytest.mark.parametrize("M", range(1, 6))
    def test_unitary(self, M, tol):
        """
        test that the outputs agree with the interferometer class when
        transformation is unitary
        """
        a = np.arange(4 * M**2, dtype=np.float64).reshape((2 * M, 2 * M))
        cov = a @ a.T + np.eye(2 * M)
        mu = np.arange(2 * M, dtype=np.float64)

        U = M ** (-0.5) * np.fft.fft(np.eye(M))
        S_U = symplectic.interferometer(U)
        cov_U = S_U @ cov @ S_U.T
        mu_U = S_U @ mu

        mu_T, cov_T = symplectic.passive_transformation(mu, cov, U)

        assert np.allclose(mu_U, mu_T, atol=tol, rtol=0)
        assert np.allclose(cov_U, cov_T, atol=tol, rtol=0)

    @pytest.mark.parametrize("hbar", [1, 2, 1.05e-34])
    def test_hbar(self, hbar, tol):
        """test that the output is a valid covariance matrix, even when not square"""

        M = 4
        a = np.arange(4 * M**2, dtype=np.float64).reshape((2 * M, 2 * M))
        cov = a @ a.T + np.eye(2 * M)
        mu = np.arange(2 * M, dtype=np.float64)

        T = np.sqrt(0.9) * M ** (-0.5) * np.ones((6, M), dtype=np.float64)

        _, cov_out = symplectic.passive_transformation(mu, cov, T, hbar=hbar)

        assert is_valid_cov(cov_out, hbar=hbar, atol=tol, rtol=0)


class TestReducedState:
    """Tests for the reduced state function"""

    def test_exception(
        self,
    ):
        """Test error is raised if requesting a non-existant subsystem"""
        with pytest.raises(ValueError, match="cannot be larger than number of subsystems"):
            symplectic.reduced_state(np.array([0, 0]), np.identity(2), [6, 4])

    def test_integer(self, hbar, tol):
        """Test requesting via an integer"""
        mu, cov = symplectic.vacuum_state(4, hbar=hbar)
        res = symplectic.reduced_state(mu, cov, 0)
        expected = np.zeros([2]), np.identity(2) * hbar / 2

        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_all(self, hbar, tol):
        """Test requesting all wires returns the full state"""
        mu, cov = symplectic.vacuum_state(4, hbar=hbar)
        res = symplectic.reduced_state(mu, cov, [0, 1, 2, 3])
        expected = np.zeros([8]), np.identity(8) * hbar / 2

        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_tms(self, hbar, tol):
        """Test reduced state of a TMS state is a thermal state"""
        r = 0.543
        phi = 0.432

        S = symplectic.two_mode_squeezing(r, phi)
        mu = np.zeros([4])
        cov = S @ S.T * (hbar / 2)

        res = symplectic.reduced_state(mu, cov, 0)

        # expected state
        nbar = np.sinh(r) ** 2
        expected = np.zeros([2]), np.identity(2) * (2 * nbar + 1) * hbar / 2

        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)


class TestLossChannel:
    """Tests for the loss channel"""

    def test_TMS_against_interferometer(self, hbar, tol):
        """Test that the loss channel on a TMS state corresponds to a beamsplitter
        acting on the mode with loss and an ancilla vacuum state"""
        r = 0.543
        phi = 0.432
        T = 0.812

        S = symplectic.two_mode_squeezing(r, phi)
        cov = S @ S.T * (hbar / 2)

        # perform loss
        _, cov_res = symplectic.loss(np.zeros([4]), cov, T, mode=0, hbar=hbar)

        # create a two mode beamsplitter acting on modes 0 and 2
        B = np.array(
            [
                [np.sqrt(T), -np.sqrt(1 - T), 0, 0],
                [np.sqrt(1 - T), np.sqrt(T), 0, 0],
                [0, 0, np.sqrt(T), -np.sqrt(1 - T)],
                [0, 0, np.sqrt(1 - T), np.sqrt(T)],
            ]
        )

        B = symplectic.expand(B, modes=[0, 2], N=3)

        # add an ancilla vacuum state in mode 2
        cov_expand = np.identity(6) * hbar / 2
        cov_expand[:2, :2] = cov[:2, :2]
        cov_expand[3:5, :2] = cov[2:, :2]
        cov_expand[:2, 3:5] = cov[:2, 2:]
        cov_expand[3:5, 3:5] = cov[2:, 2:]

        # apply the beamsplitter to modes 0 and 2
        cov_expand = B @ cov_expand @ B.T

        # compare loss function result to an interferometer mixing mode 0 with the vacuum
        _, cov_expected = symplectic.reduced_state(np.zeros([6]), cov_expand, modes=[0, 1])
        assert np.allclose(cov_expected, cov_res, atol=tol, rtol=0)

    def test_displaced_loss_against_interferometer(self, hbar, tol):
        """Test that the loss channel on a displaced state corresponds to a beamsplitter
        acting on the mode with loss and an ancilla vacuum state"""
        T = 0.812

        alpha = np.random.random(size=[2]) + np.random.random(size=[2]) * 1j
        mu = np.concatenate([alpha.real, alpha.imag])

        # perform loss
        mu_res, _ = symplectic.loss(mu, np.identity(4), T, mode=0, hbar=hbar)

        # create a two mode beamsplitter acting on modes 0 and 2
        B = np.array(
            [
                [np.sqrt(T), -np.sqrt(1 - T), 0, 0],
                [np.sqrt(1 - T), np.sqrt(T), 0, 0],
                [0, 0, np.sqrt(T), -np.sqrt(1 - T)],
                [0, 0, np.sqrt(1 - T), np.sqrt(T)],
            ]
        )

        B = symplectic.expand(B, modes=[0, 2], N=3)

        # apply the beamsplitter to modes 0 and 2
        mu_expand = np.zeros([6])
        mu_expand[np.array([0, 1, 3, 4])] = mu
        mu_expected, _ = symplectic.reduced_state(B @ mu_expand, np.identity(6), modes=[0, 1])

        # compare loss function result to an interferometer mixing mode 0 with the vacuum
        assert np.allclose(mu_expected, mu_res, atol=tol, rtol=0)

    def test_loss_complete(self, hbar, tol):
        """Test full loss on half a TMS"""
        r = 0.543
        phi = 0.432
        T = 0

        S = symplectic.two_mode_squeezing(r, phi)
        mu = np.zeros([4])
        cov = S @ S.T * (hbar / 2)

        mu, cov = symplectic.loss(mu, cov, T, mode=0, hbar=hbar)

        # expected state mode 0
        expected0 = np.zeros([2]), np.identity(2) * hbar / 2
        res0 = symplectic.reduced_state(mu, cov, 0)

        # expected state mode 1
        nbar = np.sinh(r) ** 2
        expected1 = np.zeros([2]), np.identity(2) * (2 * nbar + 1) * hbar / 2
        res1 = symplectic.reduced_state(mu, cov, 1)

        assert np.allclose(res0[0], expected0[0], atol=tol, rtol=0)
        assert np.allclose(res0[1], expected0[1], atol=tol, rtol=0)

        assert np.allclose(res1[0], expected1[0], atol=tol, rtol=0)
        assert np.allclose(res1[1], expected1[1], atol=tol, rtol=0)

    def test_loss_none(self, hbar, tol):
        """Test no loss on half a TMS leaves state unchanged"""
        r = 0.543
        phi = 0.432
        T = 1

        S = symplectic.two_mode_squeezing(r, phi)
        mu = np.zeros([4])
        cov = S @ S.T * (hbar / 2)

        res = symplectic.loss(mu, cov, T, mode=0, hbar=hbar)
        expected = mu, cov

        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_loss_thermal_state(self, hbar, tol):
        """Test loss on part of a thermal state"""
        nbar = np.array([0.4532, 0.123, 0.432])
        T = 0.54

        mu = np.zeros([2 * len(nbar)])
        cov = np.diag(2 * np.tile(nbar, 2) + 1) * (hbar / 2)

        res = symplectic.loss(mu, cov, T, mode=1, hbar=hbar)

        # the loss reduces the fractional mean photon number of mode 1
        new_nbar = nbar * np.array([1, T, 1])
        expected = mu, np.diag(2 * np.tile(new_nbar, 2) + 1) * (hbar / 2)

        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_loss_complete_random(self, hbar, tol):
        """Test loss on random state"""
        T = 0

        mu = np.random.random(size=[4])
        cov = np.array(
            [
                [10.4894171, 4.44832813, 7.35223928, -14.0593551],
                [4.44832813, 5.29244335, -1.48437419, -4.79381772],
                [7.35223928, -1.48437419, 11.92921345, -11.47687254],
                [-14.0593551, -4.79381772, -11.47687254, 19.67522694],
            ]
        )

        res = symplectic.loss(mu, cov, T, mode=0, hbar=hbar)

        # the loss reduces the fractional mean photon number of mode 1
        mu_exp = mu.copy()
        mu_exp[0] = 0
        mu_exp[2] = 0
        cov_exp = np.array(
            [
                [hbar / 2, 0, 0, 0],
                [0, 5.29244335, 0, -4.79381772],
                [0, 0, hbar / 2, 0],
                [0, -4.79381772, 0, 19.67522694],
            ]
        )

        assert np.allclose(res[1], cov_exp, atol=tol, rtol=0)
        assert np.allclose(res[0], mu_exp, atol=tol, rtol=0)


class TestMeanPhotonNumber:
    """Tests for the mean photon number function"""

    def test_coherent(self, hbar, tol):
        """Test that E(n) = |a|^2 and var(n) = |a|^2 for a coherent state"""
        a = 0.23 + 0.12j
        mu = np.array([a.real, a.imag]) * np.sqrt(2 * hbar)
        cov = np.identity(2) * hbar / 2

        mean_photon, var = symplectic.mean_photon_number(mu, cov, hbar=hbar)

        assert np.allclose(mean_photon, np.abs(a) ** 2, atol=tol, rtol=0)
        assert np.allclose(var, np.abs(a) ** 2, atol=tol, rtol=0)

    def test_squeezed(self, hbar, tol):
        """Test that E(n)=sinh^2(r) and var(n)=2(sinh^2(r)+sinh^4(r)) for a squeezed state"""
        r = 0.1
        phi = 0.423

        S = np.array(
            [
                [np.cosh(r) - np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
                [-np.sin(phi) * np.sinh(r), np.cosh(r) + np.cos(phi) * np.sinh(r)],
            ]
        )

        mu = np.zeros([2])
        cov = S @ S.T * hbar / 2

        mean_photon, var = symplectic.mean_photon_number(mu, cov, hbar=hbar)

        assert np.allclose(mean_photon, np.sinh(r) ** 2, atol=tol, rtol=0)
        assert np.allclose(var, 2 * (np.sinh(r) ** 2 + np.sinh(r) ** 4), atol=tol, rtol=0)

    def test_displaced_squeezed(self, hbar, tol):
        """Test that E(n) = sinh^2(r)+|a|^2 for a displaced squeezed state"""
        a = 0.12 - 0.05j
        r = 0.1
        phi = 0.423

        S = np.array(
            [
                [np.cosh(r) - np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
                [-np.sin(phi) * np.sinh(r), np.cosh(r) + np.cos(phi) * np.sinh(r)],
            ]
        )

        mu = np.array([a.real, a.imag]) * np.sqrt(2 * hbar)
        cov = S @ S.T * hbar / 2

        mean_photon, _ = symplectic.mean_photon_number(mu, cov, hbar=hbar)

        mean_ex = np.abs(a) ** 2 + np.sinh(r) ** 2
        assert np.allclose(mean_photon, mean_ex, atol=tol, rtol=0)

    def test_displaced_thermal(self, hbar, tol):
        """Test that E(n)=|a|^2+nbar and var(n)=var_th+|a|^2(1+2nbar)"""

        a = 0.12 - 0.05j
        nbar = 0.123

        mu = np.array([a.real, a.imag]) * np.sqrt(2 * hbar)
        cov = np.diag(2 * np.tile(nbar, 2) + 1) * (hbar / 2)

        mean_photon, var = symplectic.mean_photon_number(mu, cov, hbar=hbar)

        mean_ex = np.abs(a) ** 2 + nbar
        var_ex = nbar**2 + nbar + np.abs(a) ** 2 * (1 + 2 * nbar)

        assert np.allclose(mean_photon, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)


# pylint: disable=too-few-public-methods
class TestVectorExpansion:
    """Tests for expanding a displacement operation into a phase-space displacement vector"""

    def test_expand_one(self, hbar, tol):
        """Test that displacement vectors are created correctly"""
        alpha = 1.4 + 3.7 * 1j
        mode = 4
        N = 10
        r = symplectic.expand_vector(alpha, mode, N, hbar)
        expected = np.zeros([2 * N])
        expected[mode] = np.sqrt(2 * hbar) * alpha.real
        expected[mode + N] = np.sqrt(2 * hbar) * alpha.imag
        assert np.allclose(r, expected, atol=tol, rtol=0)


class TestExpandPassive:
    """Tests for expanding a displacement operation into a phase-space displacement vector"""

    def test_expand_one(self, tol):
        """Test that a 1x1 matrix is expanded correctly"""
        T = np.array([[0.5]])

        T_expand = symplectic.expand_passive(T, [1], 3)

        expected = np.array([[1, 0, 0], [0, 0.5, 0], [0, 0, 1]])

        assert np.allclose(T_expand, expected, atol=tol, rtol=0)

    def test_expend_not_square(self):
        """test that error is raised for non square input"""
        with pytest.raises(ValueError, match="The input matrix is not square"):
            symplectic.expand_passive(np.ones((3, 2)), [0, 1, 2], 5)

    def test_modes_length(self):
        """test that error is raised when length of modes array is incorrect"""
        with pytest.raises(ValueError, match="length of modes must match the shape of T"):
            symplectic.expand_passive(np.ones((3, 3)), [0, 1, 2, 3, 4], 8)


@pytest.mark.parametrize(
    "matrix_type",
    [np.array, csc_array, csr_array, bsr_array, lil_array, dok_array, coo_array, dia_array],
)
class TestSymplecticExpansion:
    """Tests for the expanding a symplectic matrix"""

    @pytest.mark.parametrize("mode", range(3))
    def test_expand_one(self, mode, matrix_type, tol):
        """Test expanding a one mode gate"""
        r = 0.1
        phi = 0.423
        N = 3

        S = np.array(
            [
                [np.cosh(r) - np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
                [-np.sin(phi) * np.sinh(r), np.cosh(r) + np.cos(phi) * np.sinh(r)],
            ]
        )
        S = matrix_type(S, dtype=S.dtype)

        res = symplectic.expand(S, modes=mode, N=N)
        if issparse(S):
            S, res = S.toarray(), res.toarray()

        expected = np.identity(2 * N)
        expected[mode, mode] = S[0, 0]
        expected[mode, mode + N] = S[0, 1]
        expected[mode + N, mode] = S[1, 0]
        expected[mode + N, mode + N] = S[1, 1]

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("m1, m2", [[0, 1], [0, 2], [1, 2], [2, 1]])
    def test_expand_two(self, m1, m2, matrix_type, tol):
        """Test expanding a two mode gate"""
        r = 0.1
        phi = 0.423
        N = 4

        S = symplectic.two_mode_squeezing(r, phi)
        S = matrix_type(S, dtype=S.dtype)

        res = symplectic.expand(S, modes=[m1, m2], N=N)
        if issparse(S):
            S, res = S.toarray(), res.toarray()

        expected = np.identity(2 * N)

        # mode1 terms
        expected[m1, m1] = S[0, 0]
        expected[m1, m1 + N] = S[0, 2]
        expected[m1 + N, m1] = S[2, 0]
        expected[m1 + N, m1 + N] = S[2, 2]

        # mode2 terms
        expected[m2, m2] = S[1, 1]
        expected[m2, m2 + N] = S[1, 3]
        expected[m2 + N, m2] = S[3, 1]
        expected[m2 + N, m2 + N] = S[3, 3]

        # cross terms
        expected[m1, m2] = S[0, 1]
        expected[m1, m2 + N] = S[0, 3]
        expected[m1 + N, m2] = S[2, 1]
        expected[m1 + N, m2 + N] = S[2, 3]

        expected[m2, m1] = S[1, 0]
        expected[m2, m1 + N] = S[3, 0]
        expected[m2 + N, m1] = S[1, 2]
        expected[m2 + N, m1 + N] = S[3, 2]

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("N", range(1, 6))
    def test_extend_single_mode_symplectic(self, N, matrix_type, tol):
        """Test that passing a single mode symplectic along with many modes
        makes the gate act on those modes."""

        modes = np.random.choice(N, N - 1, replace=False)
        S = random_symplectic(1)
        S = matrix_type(S, dtype=S.dtype)

        res = symplectic.expand(S, modes=modes, N=N)

        if issparse(S):
            S = S.toarray()
        for m in range(N):
            if m in modes:
                # check the symplectic acts on the mode m
                assert np.allclose(res[m, m], S[0, 0], atol=tol, rtol=0)  # X
                assert np.allclose(res[m + N, m + N], S[1, 1], atol=tol, rtol=0)  # P
                assert np.allclose(res[m, m + N], S[0, 1], atol=tol, rtol=0)  # XP
                assert np.allclose(res[m + N, m], S[1, 0], atol=tol, rtol=0)  # PX
            else:
                # check the identity acts on the mode m
                assert np.allclose(res[m, m], 1, atol=tol, rtol=0)  # X
                assert np.allclose(res[m + N, m + N], 1, atol=tol, rtol=0)  # P
                assert np.allclose(res[m, m + N], 0, atol=tol, rtol=0)  # XP
                assert np.allclose(res[m + N, m], 0, atol=tol, rtol=0)  # PX


class TestIntegration:
    """Integration tests"""

    def test_inverse_ops_cancel(self, hbar, tol):
        """Test that applying squeezing and interferometers to a four mode circuit,
        followed by applying the inverse operations, return the state to the vacuum"""

        # the symplectic matrix
        O = np.block([[np.zeros([4, 4]), np.identity(4)], [-np.identity(4), np.zeros([4, 4])]])

        # begin in the vacuum state
        mu_init, cov_init = symplectic.vacuum_state(4, hbar=hbar)

        # add displacement
        alpha = np.random.random(size=[4]) + np.random.random(size=[4]) * 1j
        D = np.concatenate([alpha.real, alpha.imag])
        mu = mu_init + D
        cov = cov_init.copy()

        # random squeezing
        r = np.random.random()
        phi = np.random.random()
        S = symplectic.expand(symplectic.two_mode_squeezing(r, phi), modes=[0, 1], N=4)

        # check symplectic
        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)

        # random interferometer
        # fmt:off
        u = np.array([[-0.06658906-0.36413058j, 0.07229868+0.65935896j, 0.59094625-0.17369183j, -0.18254686-0.10140904j],
                      [0.53854866+0.36529723j, 0.61152793+0.15022026j, 0.05073631+0.32624882j, -0.17482023-0.20103772j],
                      [0.34818923+0.51864844j, -0.24334624+0.0233729j, 0.3625974 -0.4034224j, 0.10989667+0.49366039j],
                      [0.16548085+0.14792642j, -0.3012549 -0.11387682j, -0.12731847-0.44851389j, -0.55816075-0.5639976j]])
        # fmt:on
        U = symplectic.interferometer(u)

        # check unitary
        assert np.allclose(u @ u.conj().T, np.identity(4), atol=tol, rtol=0)
        # check symplectic
        assert np.allclose(U @ O @ U.T, O, atol=tol, rtol=0)

        # apply squeezing and interferometer
        cov = U @ S @ cov @ S.T @ U.T
        mu = U @ S @ mu

        # check we are no longer in the vacuum state
        assert not np.allclose(mu, mu_init, atol=tol, rtol=0)
        assert not np.allclose(cov, cov_init, atol=tol, rtol=0)

        # return the inverse operations
        Sinv = symplectic.expand(symplectic.two_mode_squeezing(-r, phi), modes=[0, 1], N=4)
        Uinv = symplectic.interferometer(u.conj().T)

        # check inverses
        assert np.allclose(Uinv, np.linalg.inv(U), atol=tol, rtol=0)
        assert np.allclose(Sinv, np.linalg.inv(S), atol=tol, rtol=0)

        # apply the inverse operations
        cov = Sinv @ Uinv @ cov @ Uinv.T @ Sinv.T
        mu = Sinv @ Uinv @ mu

        # inverse displacement
        mu -= D

        # check that we return to the vacuum state
        assert np.allclose(mu, mu_init, atol=tol, rtol=0)
        assert np.allclose(cov, cov_init, atol=tol, rtol=0)


def test_is_symplectic():
    """Tests that the matrices generated in the symplectic module are indeed symplectic"""
    theta = np.pi / 6
    r = np.arcsinh(1.0)
    phi = np.pi / 8
    S = symplectic.rotation(theta)
    assert symplectic.is_symplectic(S)
    S = symplectic.squeezing(r, theta)
    assert symplectic.is_symplectic(S)
    S = symplectic.beam_splitter(theta, phi)
    assert symplectic.is_symplectic(S)
    S = symplectic.two_mode_squeezing(r, theta)
    assert symplectic.is_symplectic(S)
    A = np.array([[2.0, 3.0], [4.0, 6.0]])
    assert not symplectic.is_symplectic(A)
    A = np.identity(3)
    assert not symplectic.is_symplectic(A)
    A = np.array([[2.0, 3.0], [4.0, 6.0], [4.0, 6.0]])
    assert not symplectic.is_symplectic(A)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_sympmat(n):
    """test X_n = [[0, I], [-I, 0]]"""
    I = np.identity(n)
    O = np.zeros_like(I)
    X = np.block([[O, I], [-I, O]])
    res = symplectic.sympmat(n)
    assert np.all(X == res)


@pytest.mark.parametrize("n", [5, 10, 50])
@pytest.mark.parametrize("datatype", [np.complex128, np.float64])
@pytest.mark.parametrize("svd_order", [True, False])
def test_autonne(n, datatype, svd_order):
    """Checks the correctness of the Autonne decomposition function"""
    if datatype is np.complex128:
        A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    if datatype is np.float64:
        A = np.random.rand(n, n)
    A += A.T
    r, U = symplectic.autonne(A, svd_order=svd_order)
    assert np.allclose(A, U @ np.diag(r) @ U.T)
    assert np.all(r >= 0)
    if svd_order is True:
        assert np.all(np.diff(r) <= 0)
    else:
        assert np.all(np.diff(r) >= 0)


def test_autonne_error():
    """Tests the value errors of autonne"""
    n = 10
    m = 20
    A = np.random.rand(n, m)
    with pytest.raises(ValueError, match="The input matrix is not square"):
        symplectic.autonne(A)
    n = 10
    m = 10
    A = np.random.rand(n, m)
    with pytest.raises(ValueError, match="The input matrix is not symmetric"):
        symplectic.autonne(A)


class TestPhaseSpaceFunctions:
    """Tests for the shared phase space operations"""

    def test_means_changebasis(self):
        """Test the change of basis function applied to vectors. This function
        converts from xp to symmetric ordering, and vice versa."""
        means_xp = np.array([1, 2, 3, 4, 5, 6])
        means_symmetric = np.array([1, 4, 2, 5, 3, 6])

        assert np.all(symplectic.xxpp_to_xpxp(means_xp) == means_symmetric)
        assert np.all(symplectic.xpxp_to_xxpp(means_symmetric) == means_xp)

    def test_cov_changebasis(self):
        """Test the change of basis function applied to matrices. This function
        converts from xp to symmetric ordering, and vice versa."""
        cov_xp = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])

        cov_symmetric = np.array([[0, 2, 1, 3], [8, 10, 9, 11], [4, 6, 5, 7], [12, 14, 13, 15]])

        assert np.all(symplectic.xxpp_to_xpxp(cov_xp) == cov_symmetric)
        assert np.all(symplectic.xpxp_to_xxpp(cov_symmetric) == cov_xp)

    @pytest.mark.parametrize("fun", [symplectic.xxpp_to_xpxp, symplectic.xpxp_to_xxpp])
    def test_change_basis_raises_not_square(self, fun):
        """Test correct error is raised when a non-square matrix is passed"""
        A = np.random.rand(4, 6)
        with pytest.raises(ValueError, match="The input matrix is not square"):
            fun(A)

    @pytest.mark.parametrize("fun", [symplectic.xxpp_to_xpxp, symplectic.xpxp_to_xxpp])
    @pytest.mark.parametrize("dim", [1, 2])
    def test_change_basis_raises_not_even(self, fun, dim):
        """Test correct error is raised when a non-even-dimensional array is passed"""
        size = (5,) * dim
        A = np.random.rand(*size)
        with pytest.raises(ValueError, match="The input array is not even-dimensional"):
            fun(A)

    @pytest.mark.parametrize("dim", [2, 4, 6, 8])
    def test_functional_inverse(self, dim):
        """Check that xxpp_to_xpxp is the inverse of xpxp_to_xxpp and viceversa"""
        M = np.random.rand(dim, dim)
        assert np.all(M == symplectic.xxpp_to_xpxp(symplectic.xpxp_to_xxpp(M)))
        assert np.all(M == symplectic.xpxp_to_xxpp(symplectic.xxpp_to_xpxp(M)))

        v = np.random.rand(dim)
        assert np.all(v == symplectic.xxpp_to_xpxp(symplectic.xpxp_to_xxpp(v)))
        assert np.all(v == symplectic.xpxp_to_xxpp(symplectic.xxpp_to_xpxp(v)))
