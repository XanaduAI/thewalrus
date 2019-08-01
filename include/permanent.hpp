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
#include <numeric>
#include <random>
#include "fsum.hpp"

typedef unsigned long long int ullint;
typedef long long int llint;
//typedef long double qp;

#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
typedef __float128 qp;
//#include <quadmath.h>
#else
typedef long double qp;
#endif

/**
 * Gray code generator.
 *
 * @param n
 * @return the corresponding Gray code
 */
llint igray(llint n)
{
    /* Right Shift the number by 1
       taking xor with original number */
    return n ^ (n >> 1);
}

/**
 * Returns left most set bit.
 *
 * @param n
 * @return the left most set bit
 */
int left_most_set_bit(llint n) {

    if ( n == 0)
        return 0;

    int msb = 0;

    while ( n != 0) {
        n = n / 2;
        msb++;
    }

    return msb;
}


/**
 * Decimal to binary conversion
 *
 * @param \f$k\f$
 * @param \f$n\f$
 * @return \f$n\f$ bit binary representation of integer \f$k\f$
 */
std::vector<int> dec2bin(llint &k, int &n) {

    llint kk = k;
    int i = n;
    std::vector<int> mat(n, 0);

    while ( kk > 0 && i > 0 ) {
        mat[i-1] = kk % 2;
        kk = kk/2;
        i = i-1;
    }
    return mat;
}



namespace libwalrus {

/**
 * Returns the permanent of an matrix.
 *
 * \rst
 *
 * Returns the permanent of a matrix using Ryser's algorithm with Gray code ordering.
 *
 * \endrst
 *
 *
 * @param mat  a flattened vector of size \f$n^2\f$, representing an
 *      \f$n\times n\f$ row-ordered symmetric matrix.
 * @return permanent of the input matrix
 */
template <typename T>
inline T permanent(std::vector<T> &mat) {
    int n = std::sqrt(static_cast<double>(mat.size()));
    llint x = static_cast<llint>(pow(2,n) - 1) ;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);
#else
    int nthreads = 1;
#endif

    std::vector<T> tot(nthreads, static_cast<T>(0));

    std::vector<llint> threadbound_low(nthreads);
    std::vector<llint> threadbound_hi(nthreads);

    for (int i=0; i < nthreads; i++) {

        threadbound_low[i] = i*x/nthreads;
        threadbound_hi[i] = (i+1)*x/nthreads;
    }
    threadbound_hi[nthreads-1] = x;

    #pragma omp parallel for shared(tot)
    for (int ii = 0; ii < nthreads; ii++) {
        T permtmp = static_cast<T>(0);
        int cntr = 0;
        std::vector<int> chitmp(n, 0);
        std::vector<T> tmp(n, static_cast<T>(0));

        for (llint k = threadbound_low[ii]; k < threadbound_hi[ii]; k++) {
            T rowsumprod = static_cast<T>(1);
            llint kg2 = igray(k+1);
            llint  sgntmp = kg2 - igray(k);
            llint  sig = sgntmp/std::abs(sgntmp);
            int  pos = 0;

            pos = n - left_most_set_bit(sgntmp);

            if ( k == threadbound_low[ii] ) {
                chitmp = dec2bin(kg2, n);

                for ( int j = 0; j < n; j++) {
                    T localsum = static_cast<T>(0);
                    for (int id = 0; id < n; id++) {
                        localsum += static_cast<T>(chitmp[id]) * mat[id*n+j];
                    }
                    tmp[j] += localsum;
                    rowsumprod *= tmp[j];
                }

                cntr = static_cast<int>(std::accumulate(chitmp.begin(), chitmp.end(), 0));
            }
            else {
                cntr += sig;

                for(int j = 0; j < n; j++ ) {
                    if (sig < 0)
                        tmp[j] -= mat[pos * n + j];
                    else
                        tmp[j] += mat[pos * n + j];

                    rowsumprod *= tmp[j];
                }
            }

            if ( (static_cast<llint>(n)-cntr) % 2 == 0)
                permtmp += rowsumprod;
            else
                permtmp -= rowsumprod;

        }
        tot[ii] = permtmp;
    }

    return static_cast<T>(std::accumulate(tot.begin(), tot.end(), static_cast<T>(0)));
}



/**
 * Returns the permanent of an matrix using fsum.
 *
 * \rst
 *
 * Returns the permanent of a matrix using Ryser's algorithm with Gray code ordering.
 *
 * \endrst
 *
 *
 * @param mat  a flattened vector of size \f$n^2\f$, representing an
 *      \f$n\times n\f$ row-ordered symmetric matrix.
 * @return permanent of the input matrix
 */
template <typename T>
inline double perm_fsum(std::vector<T> &mat) {
    int n = std::sqrt(static_cast<double>(mat.size()));
    llint x = static_cast<llint>(pow(2,n) - 1) ;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);
#else
    int nthreads = 1;
#endif

    std::vector<T> tot(nthreads, static_cast<T>(0));

    std::vector<llint> threadbound_low(nthreads);
    std::vector<llint> threadbound_hi(nthreads);

    for (int i=0; i < nthreads; i++) {

        threadbound_low[i] = i*x/nthreads;
        threadbound_hi[i] = (i+1)*x/nthreads;
    }
    threadbound_hi[nthreads-1] = x;

    #pragma omp parallel for shared(tot)
    for (int ii = 0; ii < nthreads; ii++) {

        fsum::sc_partials permtmp; // = 0;

        int cntr = 0;
        std::vector<int> chitmp(n, 0);
        std::vector<T> tmp(n, static_cast<T>(0));

        for (llint k = threadbound_low[ii]; k < threadbound_hi[ii]; k++) {
            T rowsumprod = static_cast<T>(1);
            llint kg2 = igray(k+1);
            llint  sgntmp = kg2 - igray(k);
            llint  sig = sgntmp/std::abs(sgntmp);
            int  pos = 0;

            pos = n - left_most_set_bit(sgntmp);

            if ( k == threadbound_low[ii] ) {
                chitmp = dec2bin(kg2, n);

                for ( int j = 0; j < n; j++) {
                    fsum::sc_partials localsum; //= static_cast<T>(0);
                    for (int id = 0; id < n; id++) {
                        localsum += static_cast<T>(chitmp[id]) * mat[id*n+j];
                    }
                    tmp[j] += localsum;
                    rowsumprod *= tmp[j];
                }

                cntr = static_cast<int>(std::accumulate(chitmp.begin(), chitmp.end(), 0));
            }
            else {
                cntr += sig;

                for(int j = 0; j < n; j++ ) {
                    if (sig < 0)
                        tmp[j] -= mat[pos * n + j];
                    else
                        tmp[j] += mat[pos * n + j];

                    rowsumprod *= tmp[j];
                }
            }

            if ( (static_cast<llint>(n)-cntr) % 2 == 0)
                permtmp += rowsumprod;
            else
                permtmp -= rowsumprod;

        }
        tot[ii] = permtmp;
    }
    return static_cast<T>(std::accumulate(tot.begin(), tot.end(), static_cast<T>(0)));
}


/**
 * \rst
 *
 * Returns the permanent of a matrix using the Ryser's algo with Gray code ordering
 *
 * \endrst
 *
 *
 * This is a wrapper around the templated function `libwalrus::permanent` for Python
 * integration. It accepts and returns complex double numeric types, and
 * returns sensible values for empty and non-even matrices.
 *
 * In addition, this wrapper function automatically casts all matrices
 * to type `complex<long double>`, allowing for greater precision than supported
 * by Python and NumPy.
 *
 * @param mat vector representing the flattened matrix
 * @return the permanent
 */
std::complex<double> permanent_quad(std::vector<std::complex<double>> &mat) {
    std::vector<std::complex<long double>> matq(mat.begin(), mat.end());
    std::complex<long double> perm = permanent(matq);
    return static_cast<std::complex<double>>(perm);
}

/**
 * \rst
 *
 * Returns the permanent of a matrix using Ryser's algo with Gray code ordering
 *
 * \endrst
 *
 *
 * This is a wrapper around the templated function `libwalrus::permanent` for Python
 * integration. It accepts and returns double numeric types, and
 * returns sensible values for empty and non-even matrices.
 *
 * In addition, this wrapper function automatically casts all matrices
 * to type `long double`, allowing for greater precision than supported
 * by Python and NumPy.
 *
 * @param mat vector representing the flattened matrix
 * @return the permanent
 */
double permanent_quad(std::vector<double> &mat) {
    std::vector<qp> matq(mat.begin(), mat.end());
    qp perm = permanent(matq);
    return static_cast<double>(perm);
}



/**
 * \rst
 *
 * Returns the permanent of a matrix using Ryser's algo with Gray code ordering with fsum
 *
 * \endrst
 *
 *
 * This is a wrapper around the templated function `libwalrus::perm_fsum` for Python
 * integration. It accepts and returns double numeric types, and
 * returns sensible values for empty and non-even matrices.
 *
 *
 * @param mat vector representing the flattened matrix
 * @return the permanent
 */
double permanent_fsum(std::vector<double> &mat) {
    std::vector<double> matq(mat.begin(), mat.end());
    double perm = perm_fsum(matq);
    return static_cast<double>(perm);
}

}
