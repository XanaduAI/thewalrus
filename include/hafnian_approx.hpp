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
 * Contains functions for approximating the hafnian of a matrix
 * in a classically efficient manner, for certains classes of matrices.
 */

#pragma once
#include <stdafx.h>
#include <numeric>
#include <random>
#include <cmath>

#ifdef LAPACKE
#define EIGEN_SUPERLU_SUPPORT
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#endif

#include <Eigen/Eigenvalues>


namespace libwalrus {

/**
* Returns the approximation to the hafnian of a matrix with non-negative entries.
*
* The approximation follows an stochastic algorithm according to which the hafnian
* can be approximated as the sum of determinants of matrices.
* The accuracy of the approximation increases with increasing number of iterations.
*
* @param mat vector representing the flattened matrix
* @param nsamples positive integer representing the number of samples to perform
* @return the approximate hafnian
*/
template <typename T>
inline long double hafnian_nonneg(std::vector<T> &mat, int &nsamples) {
    int n = std::sqrt(static_cast<double>(mat.size()));

    long double mean = 0;
    long double stdev = 1;

    namespace eg = Eigen;
    eg::Matrix<T, eg::Dynamic, eg::Dynamic> A = eg::Map<eg::Matrix<T, eg::Dynamic, eg::Dynamic>, eg::Unaligned>(mat.data(), n, n);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);
#else
    int nthreads = 1;
#endif

    std::vector<int> threadbound_low(nthreads);
    std::vector<int> threadbound_hi(nthreads);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stdev);

    std::vector<long double> determinants(nsamples);

    #pragma omp parallel for shared(determinants)
    for (int k = 0; k < nsamples; k++) {
        std::vector<T> matrand(n * n, 0);
        std::vector<T> g(n * n, 0);
        std::vector<T> gt(n * n, 0);
        eg::Matrix<T, eg::Dynamic, eg::Dynamic> W;
        W.resize(n, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                long double randnum = distribution(generator);
                matrand[i * n + j] = static_cast<T>(randnum);
                g[i * n + j] = 0.0;
                gt[i * n + j] = 0.0;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                g[i * n + j] = matrand[i * n + j];
                gt[j * n + i] = matrand[i * n + j];
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int id = i * n + j;
                W(i, j) = (g[id] - gt[id]) * std::sqrt(std::abs(A(i, j)));
            }
        }

        long double det = std::real(W.determinant());

        determinants[k] = det;
    }

    long double final = 0.0;

    for (int i = 0; i < nsamples; i++) {
        final += determinants[i];
    }

    final = final / (static_cast<long double>(nsamples));

    return final;

}

/**
* Returns the approximation to the hafnian of a matrix with non-negative entries.
*
* The approximation follows an stochastic algorithm according to which the hafnian
* can be approximated as the sum of determinants of matrices.
* The accuracy of the approximation increases with increasing number of iterations.
*
* This is a wrapper around the templated function `libwalrus::hafnian_nonneg` for Python
* integration. It accepts and returns double numeric types, and
* returns sensible values for empty and non-even matrices.
*
* In addition, this wrapper function automatically casts all matrices
* to type `long double`, allowing for greater precision than supported
* by Python and NumPy.
*
* @param mat vector representing the flattened matrix
* @param nsamples positive integer representing the number of samples to perform
* @return the approximate hafnian
*/
double hafnian_approx(std::vector<double> &mat, int &nsamples) {
    std::vector<long double> matq(mat.begin(), mat.end());
    int n = std::sqrt(static_cast<double>(mat.size()));
    long double haf;

    if (n == 0)
        haf = 1.0;
    else if (n % 2 != 0)
        haf = 0.0;
    else
        haf = hafnian_nonneg(matq, nsamples);

    return static_cast<double>(haf);
}

}
