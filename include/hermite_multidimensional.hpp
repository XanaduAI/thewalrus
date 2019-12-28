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
/**
 * @file
 * Contains functions for calculating the multidimensional
 * Hermite polynomials, used for computation of batched hafnians.
 */

#pragma once
#include <stdafx.h>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

typedef unsigned long long int ullint;


/**
 * Returns the index of the one dimensional flattened vector corresponding to the multidimensional tensor
 *
 * @param pos
 * @param resolution
 *
 * @return index on flattened vector
 */
ullint vec2index(std::vector<int> &pos, int resolution) {
    int dim = pos.size();
    ullint nextCoordinate = 0;

    nextCoordinate = pos[0]-1;
    for(int ii = 0; ii < dim-1; ii++) {
        nextCoordinate = nextCoordinate*resolution + (pos[ii+1]-1);
    }

    return nextCoordinate;

}

namespace libwalrus {

/**
 * Returns photon number statistics of a Gaussian state for a given covariance matrix `mat`.
 * as described in *Multidimensional Hermite polynomials and photon distribution for polymode mixed light*
 * [arxiv:9308033](https://arxiv.org/abs/hep-th/9308033).
 *
 * This implementation is based on the MATLAB code available at
 * https://github.com/clementsw/gaussian-optics
 *
 * @param mat a flattened vector of size \f$2n^2\f$, representing a
 *       \f$2n\times 2n\f$ row-ordered symmetric matrix.
 * @param d a flattened vector of size \f$2n\f$, representing the first order moments.
 * @param resolution highest number of photons to be resolved.
 *
 */
template <typename T>
inline std::vector<T> hermite_multidimensional_cpp(std::vector<T> &R_mat, std::vector<T> &y_mat, int &resolution) {
    int dim = std::sqrt(static_cast<double>(R_mat.size()));

    namespace eg = Eigen;

    eg::Matrix<T, eg::Dynamic, eg::Dynamic> R = eg::Map<eg::Matrix<T, eg::Dynamic, eg::Dynamic>, eg::Unaligned>(R_mat.data(), dim, dim);
    eg::Matrix<T, eg::Dynamic, eg::Dynamic> y = eg::Map<eg::Matrix<T, eg::Dynamic, eg::Dynamic>, eg::Unaligned>(y_mat.data(), dim, dim);

    ullint Hdim = pow(resolution, dim);
    std::vector<T> H(Hdim, 0);


    H[0] = 1;

    std::vector<int> nextPos(dim, 1);
    std::vector<int> jumpFrom(dim, 1);
    std::vector<int> ek(dim, 0);
    std::vector<double> factors(resolution+1, 0);
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


        H[nextCoordinate] = H[nextCoordinate] + y(k, 0);
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







template <typename T>
inline std::vector<T> renorm_hermite_multidimensional_cpp(std::vector<T> &R_mat, std::vector<T> &y_mat, int &resolution) {
    int dim = std::sqrt(static_cast<double>(R_mat.size()));

    namespace eg = Eigen;

    eg::Matrix<T, eg::Dynamic, eg::Dynamic> R = eg::Map<eg::Matrix<T, eg::Dynamic, eg::Dynamic>, eg::Unaligned>(R_mat.data(), dim, dim);
    eg::Matrix<T, eg::Dynamic, eg::Dynamic> y = eg::Map<eg::Matrix<T, eg::Dynamic, eg::Dynamic>, eg::Unaligned>(y_mat.data(), dim, dim);

    ullint Hdim = pow(resolution, dim);
    std::vector<T> H(Hdim, 0);

    H[0] = 1;
    std::vector<double> intsqrt(resolution+1, 0);
    for (int ii = 0; ii<=resolution; ii++) {
        intsqrt[ii] = std::sqrt((static_cast<double>(ii)));
    }
    std::vector<int> nextPos(dim, 1);
    std::vector<int> jumpFrom(dim, 1);
    std::vector<int> ek(dim, 0);
    std::vector<double> factors(resolution+1, 0);
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

        H[nextCoordinate] = H[nextCoordinate] + y(k, 0)/(static_cast<double>(intsqrt[nextPos[k]-1]));
        H[nextCoordinate] = H[nextCoordinate] * H[fromCoordinate];

        std::vector<int> tmpjump(dim, 0);

        for (int ii = 0; ii < dim; ii++) {
            if (jumpFrom[ii] > 1) {
                std::vector<int> prevJump(dim, 0);
                prevJump[ii] = 1;
                std::transform(jumpFrom.begin(), jumpFrom.end(), prevJump.begin(), tmpjump.begin(), std::minus<int>());
                ullint prevCoordinate = vec2index(tmpjump, resolution);
                H[nextCoordinate] = H[nextCoordinate] - (intsqrt[jumpFrom[ii]-1]/intsqrt[nextPos[k]-1])*static_cast<T>(R(k,ii))*H[prevCoordinate];

            }
        }

    }
    return H;

}



}
