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

from thewalrus import symplectic


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
        rotation = np.array([[np.cos(phi/2), -np.sin(phi/2)], [np.sin(phi/2), np.cos(phi/2)]])
        expected = rotation @ np.diag(np.exp([-2*r, 2*r])) @ rotation.T
        assert np.allclose(out, expected, atol=tol, rtol=0)


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
        # fmt: off
        B = np.array([[1, -1, 0, 0], [1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]])/np.sqrt(2)

        Sq1 = np.array([[np.cosh(r)-np.cos(phi)*np.sinh(r), -np.sin(phi)*np.sinh(r)],
                        [-np.sin(phi)*np.sinh(r), np.cosh(r)+np.cos(phi)*np.sinh(r)]])

        Sq2 = np.array([[np.cosh(-r)-np.cos(phi)*np.sinh(-r), -np.sin(phi)*np.sinh(-r)],
                        [-np.sin(phi)*np.sinh(-r), np.cosh(-r)+np.cos(phi)*np.sinh(-r)]])
        # fmt: on

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
        expected = np.array([a1out.real, a2out.real, a1out.imag, a2out.imag]) * np.sqrt(
            2 * hbar
        )
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
        # fmt on
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


class TestReducedState:
    """Tests for the reduced state function"""

    def test_exception(self,):
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
        B = np.array([[np.sqrt(T), -np.sqrt(1-T), 0, 0],
                      [np.sqrt(1-T), np.sqrt(T), 0, 0],
                      [0, 0, np.sqrt(T), -np.sqrt(1-T)],
                      [0, 0, np.sqrt(1-T), np.sqrt(T)]])

        B = symplectic.expand(B, modes=[0, 2], N=3)

        # add an ancilla vacuum state in mode 2
        cov_expand = np.identity(6)*hbar/2
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
        B = np.array([[np.sqrt(T), -np.sqrt(1-T), 0, 0],
                      [np.sqrt(1-T), np.sqrt(T), 0, 0],
                      [0, 0, np.sqrt(T), -np.sqrt(1-T)],
                      [0, 0, np.sqrt(1-T), np.sqrt(T)]])

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
        var_ex = nbar ** 2 + nbar + np.abs(a) ** 2 * (1 + 2 * nbar)

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

class TestSymplecticExpansion:
    """Tests for the expanding a symplectic matrix"""

    @pytest.mark.parametrize("mode", range(3))
    def test_expand_one(self, mode, tol):
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

        res = symplectic.expand(S, modes=mode, N=N)

        expected = np.identity(2 * N)
        expected[mode, mode] = S[0, 0]
        expected[mode, mode + N] = S[0, 1]
        expected[mode + N, mode] = S[1, 0]
        expected[mode + N, mode + N] = S[1, 1]

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("m1, m2", [[0, 1], [0, 2], [1, 2], [2, 1]])
    def test_expand_two(self, m1, m2, tol):
        """Test expanding a two mode gate"""
        r = 0.1
        phi = 0.423
        N = 4

        S = symplectic.two_mode_squeezing(r, phi)
        res = symplectic.expand(S, modes=[m1, m2], N=N)

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
        # fmt on
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
    """ Tests that the matrices generated in the symplectic module are indeed symplectic"""
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
