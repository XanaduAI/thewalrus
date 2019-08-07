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
 * Contains functions for computing the Torontonian using the algorithm described in
 * *A faster hafnian formula for complex matrices and its benchmarking
 * on the Titan supercomputer*, [arxiv:1805.12498](https://arxiv.org/abs/1805.12498)
 */
#pragma once
#include <stdafx.h>
#include <numeric>

#ifdef LAPACKE
#define EIGEN_SUPERLU_SUPPORT
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#endif

#include <Eigen/Eigenvalues>
#include "fsum.hpp"

namespace libwalrus {
/**
 * Given a string of length `len`, finds the positions in which it has a 1
 * and stores its position i, as 2*i and 2*i+1 in consecutive slots
 * of the array pos.
 *
 * It also returns (twice) the number of ones in array dst
 *
 * @param dst character array representing binary digits.
 * @param len length of the array `dst`.
 * @param pos resulting character array of length `2*len` storing
 * the indices at which `dst` contains the values 1.
 * @return returns twice the number of ones in array `dst`.
 */
void find2T (char *dst, Byte len, Byte *pos, char offset)
{
    Byte j = offset - 1;

    for (Byte i = 0; i < len; i++) {
        if (1 == dst[i]) {
            pos[j] = len - i - 1;
            pos[j + offset] = 2 * len - i - 1;
            j--;
        }
    }
}


/**
 * Partial sum of a character array
 *
 * @param dst character array
 * @param m sum the first m characters
 *
 * @return the partial sum
 */
char sum(char *dst, Byte m) {
    char sum_tot = 0;
    for (int i = 0; i < m; i++) {
        sum_tot += (Byte)dst[i];
    }
    return sum_tot;
}

/**
 * Computes the Torontonian of an input matrix.
 *
 * If the output is NaN, that means that the input matrix does not have
 * a Torontonian with physical meaning.
 *
 * This function uses OpenMP (if available) to parallelize the reduction.
 *
 * @param mat flattened vector of size \f$n^2\f$, representing an \f$n\times n\f$
 *       row-ordered symmetric matrix.
 * @return Torontonian of the input matrix
 */
template <typename T>
inline T torontonian(std::vector<T> &mat) {
    int n = std::sqrt(static_cast<double>(mat.size()));
    Byte m = n / 2;
    unsigned long long int x = static_cast<unsigned long long int>(pow(2, m));

    namespace eg = Eigen;
    eg::Matrix<T, eg::Dynamic, eg::Dynamic> A = eg::Map<eg::Matrix<T, eg::Dynamic, eg::Dynamic>, eg::Unaligned>(mat.data(), n, n);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);
#else
    int nthreads = 1;
#endif

    std::vector<unsigned long long int> threadbound_low(nthreads);
    std::vector<unsigned long long int> threadbound_hi(nthreads);

    for (int i = 0; i < nthreads; i++) {

        threadbound_low[i] = i * x / nthreads;
        threadbound_hi[i] = (i + 1) * x / nthreads;
    }


    std::vector<T> localsum(nthreads);

    #pragma omp parallel for shared(localsum)

    for (int ii = 0; ii < nthreads; ii++) {

        T netsum = static_cast<T>(0.0);
        for (unsigned long long int k = threadbound_low[ii]; k < threadbound_hi[ii]; k++) {


            unsigned long long int xx = k;
            char* dst = new char[m];

            dec2bin(dst, xx, m);
            char len = sum(dst, m);

            Byte* short_st = new Byte[2 * len];
            find2T(dst, m, short_st, len);
            delete [] dst;

            eg::Matrix<T, eg::Dynamic, eg::Dynamic> B;
            B.resize(2 * len, 2 * len);

            for (int i = 0; i < 2 * len; i++) {
                for (int j = 0; j < 2 * len; j++) {
                    B(i, j) = -A(short_st[i], short_st[j]);
                }
            }

            delete [] short_st;

            for (int i = 0; i < 2 * len; i++) {
                B(i, i) += static_cast<T>(1);
            }

            T det = std::real(B.determinant());

            if (len % 2 == 0) {
                netsum += static_cast<T>(1.0) / std::sqrt(det);
            }
            else {
                netsum -= static_cast<T>(1.0) / std::sqrt(det);
            }

        }

        localsum[ii] = netsum;

    }

    int n_local = localsum.size();
    T final = 0.0;
    T sign = 1.0;

    if (m % 2 != 0)
        sign = -1.0;

    for (int i = 0; i < n_local; i++) {
        final += localsum[i]    ;
    }

    return sign * final;
}


/**
 * Computes the Torontonian of an input matrix using the
 * [Shewchuck algorithm](https://github.com/achan001/fsum),
 * a significantly more [accurate summation algorithm](https://link.springer.com/article/10.1007%2FPL00009321).
 *
 * Note that the fsum implementation currently only allows for
 * double precision, and precludes use of OpenMP parallelization.
 *
 * Note: if the output is NaN, that means that the input matrix does not have
 * a Torontonian with physical meaning.
 *
 * @param mat flattened vector of size \f$n^2\f$, representing an \f$n\times n\f$
 *       row-ordered symmetric matrix.
 * @return Torontonian of the input matrix
 */
template <typename T>
inline double torontonian_fsum(std::vector<T> &mat) {
    // Here weinput the matrix from python. The variable n is the size of the matrix
    int n = std::sqrt(static_cast<double>(mat.size()));
    Byte m = n / 2;
    unsigned long long int x = static_cast<unsigned long long int>(pow(2, m));

    fsum::sc_partials netsum;

    namespace eg = Eigen;
    eg::Matrix<T, eg::Dynamic, eg::Dynamic> A = eg::Map<eg::Matrix<T, eg::Dynamic, eg::Dynamic>, eg::Unaligned>(mat.data(), n, n);

    for (int k = 0; k < x; k++) {
        unsigned long long int xx = k;
        char* dst = new char[m];

        dec2bin(dst, xx, m);
        char len = sum(dst, m);

        Byte* short_st = new Byte[2 * len];
        find2T(dst, m, short_st, len);
        delete [] dst;

        // eg::Matrix<double,eg::Dynamic,eg::Dynamic> B(2*len, 2*len, 0.);
        eg::Matrix<T, eg::Dynamic, eg::Dynamic> B;
        B.resize(2 * len, 2 * len);

        for (int i = 0; i < 2 * len; i++) {
            for (int j = 0; j < 2 * len; j++) {
                B(i, j) = -A(short_st[i], short_st[j]);
            }
        }

        delete [] short_st;

        for (int i = 0; i < 2 * len; i++) {
            B(i, i) += 1;
        }

        long double det = std::real(B.determinant());

        if (len % 2 == 0) {
            netsum += 1.0 / std::sqrt(det);
        }
        else {
            netsum += -1.0 / std::sqrt(det);
        }
    }

    double sign = 1.0;

    if (m % 2 != 0)
        sign = -1.0;

    return static_cast<double>(netsum) * static_cast<double>(sign);
}


/**
 * Computes the Torontonian of an input matrix.
 *
 * If the output is NaN, that means that the input matrix does not have
 * a Torontonian with physical meaning.
 *
 * This is a wrapper around the templated function `libwalrus::torontonian` for Python
 * integration. It accepts and returns complex double numeric types, and
 * returns sensible values for empty and non-even matrices.
 *
 * In addition, this wrapper function automatically casts all matrices
 * to type `complex<long double>`, allowing for greater precision than supported
 * by Python and NumPy.
 *
 * @param mat flattened vector of size \f$n^2\f$, representing an \f$n\times n\f$
 *       row-ordered symmetric matrix.
 * @return Torontonian of the input matrix
 */
std::complex<double> torontonian_quad(std::vector<std::complex<double>> &mat) {
    std::vector<std::complex<long double>> matq(mat.begin(), mat.end());
    std::complex<long double> tor = torontonian(matq);
    return static_cast<std::complex<double>>(tor);
}


/**
 * Computes the Torontonian of an input matrix.
 *
 * If the output is NaN, that means that the input matrix does not have
 * a Torontonian with physical meaning.
 *
 * This is a wrapper around the templated function `libwalrus::torontonian` for Python
 * integration. It accepts and returns double numeric types, and
 * returns sensible values for empty and non-even matrices.
 *
 * In addition, this wrapper function automatically casts all matrices
 * to type `long double`, allowing for greater precision than supported
 * by Python and NumPy.
 *
 * @param mat flattened vector of size \f$n^2\f$, representing an \f$n\times n\f$
 *       row-ordered symmetric matrix.
 * @return Torontonian of the input matrix
 */
double torontonian_quad(std::vector<double> &mat) {
    std::vector<long double> matq(mat.begin(), mat.end());
    long double tor = torontonian(matq);
    return static_cast<double>(tor);
}

}
