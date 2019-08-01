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
 * \rst
 * Contains functions for computing the hafnian using the recursive algorithm described in
 * *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`,
 * where it is labelled as 'Algorithm 2'.
 * \endrst
 */
#pragma once
#include <stdafx.h>


namespace libwalrus {

/**
 * Recursive hafnian solver.
 *
 * Modified with permission from https://github.com/eklotek/Hafnian.
 *
 * This function uses OpenMP (if available) to parallelize the reduction.
 *
 * @param b
 * @param s
 * @param w
 * @param g
 * @param n
 * @return the hafnian
 */
template <typename T>
inline T recursive_chunk(std::vector<T> b, int s, int w, std::vector<T> g, int n) {
    if (s == 0) {
        return static_cast<T>(w) * g[n];
    }

    std::vector<T> c((s - 2) * (s - 3) / 2 * (n + 1), 0.0);
    T h, h1, h2;
    int u, v, j, k, i = 0;

    for (j = 1; j < s - 2; j++) {
        for (k = 0; k < j; k++) {
            for (u = 0; u < n + 1; u++) {
                c[(n + 1)*i + u] = b[(n + 1) * ((j + 1) * (j + 2) / 2 + k + 2) + u];
            }
            i += 1;
        }
    }

    #pragma omp task shared(h1)
    h1 = recursive_chunk(c, s - 2, -w, g, n);

    std::vector<T> e(n + 1, 0);
    e = g;

    for (u = 0; u < n; u++) {
        for (v = 0; v < n - u; v++) {
            e[u + v + 1] += g[u] * b[v];

            for (j = 1; j < s - 2; j++) {
                for (k = 0; k < j; k++) {
                    c[(n + 1) * (j * (j - 1) / 2 + k) + u + v + 1] +=
                        b[(n + 1) * ((j + 1) * (j + 2) / 2) + u]
                        * b[(n + 1) * ((k + 1) * (k + 2) / 2 + 1) + v]
                        + b[(n + 1) * (k + 1) * (k + 2) / 2 + u]
                        * b[(n + 1) * ((j + 1) * (j + 2) / 2 + 1) + v];
                }
            }
        }
    }

    #pragma omp task shared(h2)
    h2 = recursive_chunk(c, s - 2, w, e, n);

    #pragma omp taskwait
    h = h1 + h2;

    return h;
}

/**
 * Returns the hafnian of an matrix.
 *
 * \rst
 *
 * Returns the hafnian of a matrix using the recursive algorithm described in
 * *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`,
 * where it is labelled as 'Algorithm 2'.
 *
 * \endrst
 *
 * Modified with permission from https://github.com/eklotek/Hafnian.
 *
 * @param mat  a flattened vector of size \f$n^2\f$, representing an
 *      \f$n\times n\f$ row-ordered symmetric matrix.
 * @return hafnian of the input matrix
 */
template <typename T>
inline T hafnian_recursive(std::vector<T> &mat) {
    int n = std::sqrt(static_cast<double>(mat.size())) / 2;

    std::vector<T> z(n * (2 * n - 1) * (n + 1), 0);
    std::vector<T> g(n + 1, 0);

    g[0] = 1;

    #pragma omp parallel for
    for (int j = 1; j < 2 * n; j++) {
        for (int k = 0; k < j; k++) {
            z[(n + 1) * (j * (j - 1) / 2 + k)] = mat[2 * j * n + k];
        }
    }

    T result;

    #pragma omp parallel
    #pragma omp single nowait
    result = recursive_chunk(z, 2 * n, 1, g, n);

    return result;
}


/**
 * \rst
 *
 * Returns the hafnian of a matrix using the recursive algorithm described in
 * *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`,
 * where it is labelled as 'Algorithm 2'.
 *
 * \endrst
 *
 * Modified with permission from https://github.com/eklotek/Hafnian.
 *
 * This is a wrapper around the templated function `libwalrus::hafnian_recursive` for Python
 * integration. It accepts and returns complex double numeric types, and
 * returns sensible values for empty and non-even matrices.
 *
 * In addition, this wrapper function automatically casts all matrices
 * to type `complex<long double>`, allowing for greater precision than supported
 * by Python and NumPy.
 *
 * @param mat vector representing the flattened matrix
 * @return the hafnian
 */
std::complex<double> hafnian_recursive_quad(std::vector<std::complex<double>> &mat) {
    std::vector<std::complex<long double>> matq(mat.begin(), mat.end());
    int n = std::sqrt(static_cast<double>(mat.size()));
    std::complex<long double> haf;

    if (n == 0)
        haf = std::complex<double>(1.0, 0.0);
    else if (n % 2 != 0)
        haf = std::complex<double>(0.0, 0.0);
    else
        haf = hafnian_recursive(matq);

    return static_cast<std::complex<double>>(haf);
}


/**
 * \rst
 *
 * Returns the hafnian of a matrix using the recursive algorithm described in
 * *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`,
 * where it is labelled as 'Algorithm 2'.
 *
 * \endrst
 *
 * Modified with permission from https://github.com/eklotek/Hafnian.
 *
 * This is a wrapper around the templated function `libwalrus::hafnian_recursive` for Python
 * integration. It accepts and returns double numeric types, and
 * returns sensible values for empty and non-even matrices.
 *
 * In addition, this wrapper function automatically casts all matrices
 * to type `long double`, allowing for greater precision than supported
 * by Python and NumPy.
 *
 * @param mat vector representing the flattened matrix
 * @return the hafnian
 */
double hafnian_recursive_quad(std::vector<double> &mat) {
    std::vector<long double> matq(mat.begin(), mat.end());
    int n = std::sqrt(static_cast<double>(mat.size()));
    long double haf;

    if (n == 0)
        haf = 1.0;
    else if (n % 2 != 0)
        haf = 0.0;
    else
        haf = hafnian_recursive(matq);

    return static_cast<double>(haf);
}
}
