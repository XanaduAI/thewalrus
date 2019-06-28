// Copyright 2019 Xanadu Quantum Technologies Inc.


// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <stdafx.h>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

typedef unsigned long long int ullint;


ullint vec2index(std::vector<int> &pos, int resolution) {
    int dim = pos.size();
    ullint nextCoordinate = 0;

    nextCoordinate = pos[0]-1;
    for(int ii = 0; ii < dim-1; ii++) {
        nextCoordinate = nextCoordinate*resolution + (pos[ii+1]-1);
    }

    return nextCoordinate;

}

namespace hafnian {

/**
 * Returns photon numbemr statistics of a Gaussian state for a given covariance matrix `mat`.
 * Based in the MATLAB code available at: https://github.com/clementsw/gaussian-optics
 *
 * @param mat a flattened vector of size \f$2n^2\f$, representing an
 *       \f$2n\times 2n\f$ row-ordered symmetric matrix.
 * @param d a flattened vector of size \f$2n\f$, representing the first order moments.
 * @param resolution highest number of photons to be resolved.
 *
 */
template <typename T>
inline std::vector<T> mode_elem(std::vector<T> &mat, std::vector<T> &d, int &resolution) {
    int dim = std::sqrt(static_cast<double>(mat.size()));

    namespace eg = Eigen;

    eg::Matrix<T, eg::Dynamic, eg::Dynamic> M = eg::Map<eg::Matrix<T, eg::Dynamic, eg::Dynamic>, eg::Unaligned>(mat.data(), dim, dim);
    eg::Matrix<T, eg::Dynamic, eg::Dynamic> U, dim_eye, U1, U2, U3, tmp1, tmp2, tmp, tmp1_inv, tmp2_inv, R, y;
    eg::Matrix<T, eg::Dynamic, eg::Dynamic> d_eg, tmpy;

    d_eg.resize(dim, 1);
    U.resize(dim, dim);
    U1.resize(dim, dim);
    U2.resize(dim, dim);
    U3.resize(dim, dim);
    dim_eye.resize(dim, dim);

    U.setIdentity(dim, dim);
    dim_eye.setIdentity(dim, dim);

    for(int i = 0; i < dim/2; i++) {
        U(i, i) = std::complex<double>(0, -1);
        U(i+dim/2, i) = std::complex<double>(1, 0);
        U(i, i+dim/2) = std::complex<double>(0, 1);
    }



    for (int i = 0; i < dim; i++)
        d_eg(i, 0) = d[i];

    U = U/(static_cast<T>(std::sqrt(2)));


    U3 = U.transpose();
    U1 = U3.conjugate();
    U2 = U.conjugate();

    tmp1 = dim_eye + 2*M;
    tmp1_inv = tmp1.inverse();

    tmp2 = dim_eye - 2*M;
    tmp2_inv = tmp2.inverse();

    //R=U1*(eye(dim)-2*M)*inv(eye(dim)+2*M)*U2;
    tmp = tmp1_inv * U2;
    tmp = tmp2 * tmp;
    R = U1 * tmp;

    //y = 2*U3*inv(eye(dim)-2*M)*d;
    tmpy = tmp2_inv * d_eg;
    y = (U3 * tmpy);
    y = 2 * y;


    ullint Hdim = pow(resolution, dim);
    std::vector<T> H(Hdim, 0);

    H[0] = 1;

    std::vector<int> nextPos(dim, 1);
    std::vector<int> jumpFrom(dim, 1);
    std::vector<int> ek(dim, 0);
    int jump = 0;


    for (ullint jj = 0; jj < Hdim-1; jj++) {

        if (jump > 0) {
            jumpFrom[jump] += 1;
            jump = 0;
        }


        for (int ii = 0; ii < dim; ii++) {
            std::vector<int> forwardStep(dim, 0);
            forwardStep[ii] = 1;

            if ( forwardStep[ii] + nextPos[ii] > resolution) {
                nextPos[ii] = 1;
                jumpFrom[ii] = 1;
                jump = ii+1;
            }
            else {
                jumpFrom[ii] = nextPos[ii];
                nextPos[ii] = nextPos[ii] + 1;
                break;
            }
        }

        for (int ii = 0; ii < dim; ii++)
            ek[ii] = nextPos[ii] - jumpFrom[ii];

        int k = 0;
        for(; k < static_cast<int>(ek.size()); k++) {
            if(ek[k]) break;
        }

        ullint nextCoordinate = vec2index(nextPos, resolution);
        ullint fromCoordinate = vec2index(jumpFrom, resolution);


        for (int ii = 0; ii < dim; ii++) {
            H[nextCoordinate] = H[nextCoordinate] + R(k, ii) * y(ii, 0);
        }
        H[nextCoordinate] = H[nextCoordinate] * H[fromCoordinate];

        std::vector<int> tmpjump(dim, 0);

        for (int ii = 0; ii < dim; ii++) {
            if (jumpFrom[ii] > 1) {
                std::vector<int> prevJump(dim, 0);
                prevJump[ii] = 1;
                std::transform(jumpFrom.begin(), jumpFrom.end(), prevJump.begin(), tmpjump.begin(), std::minus<int>());
                ullint prevCoordinate = vec2index(tmpjump, resolution);
                H[nextCoordinate] = H[nextCoordinate] - (static_cast<T>(jumpFrom[ii]-1))*static_cast<T>(R(k,ii))*H[prevCoordinate];
            }
        }

    }


    return H;

}



/**
 * A wrapper around the templated function hafnian::mode_elem. Returns photon number
 * statistics of a Gaussian state for a given covariance matrix `mat`. Based in the
 * MATLAB code available at: https://github.com/clementsw/gaussian-optics
 *
 * @param mat a flattened vector of size \f$2n^2\f$, representing an
 *       \f$2n\times 2n\f$ row-ordered symmetric matrix.
 * @param d a flattened vector of size \f$2n\f$, representing the first order moments.
 * @param resolution highest number of photons to be resolved.
 *
 */
std::vector<std::complex<double>> batchhafnian_all(std::vector<double> &mat, std::vector<double> &d, int &resolution) {
    std::vector<std::complex<double>> matq(mat.begin(), mat.end());
    std::vector<std::complex<double>> dq(d.begin(), d.end());

    return mode_elem(matq, dq, resolution);
}


}
