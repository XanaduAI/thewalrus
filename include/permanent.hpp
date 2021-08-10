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
static inline llint igray(llint n)
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
static inline int left_most_set_bit(llint n) 
{
    if (n == 0)
        return 0;

    int msb = 0;

    while (n != 0) {
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
static inline std::vector<int> dec2bin(llint &k, int &n) 
{
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

/**
 * Get the next ordering index
 *
 * @param \f$l\f$
 * @param \f$k\f$
 * @return the \f$k+1\f$-th ordering index with updating \f$l\f$ from init index \f$k\f$
 */
static inline size_t next_perm_ordering_index(std::vector<size_t> &l, size_t k)
{
    l[0] = 0;
    l[k] = l[k+1];
    l[k+1] = k+1;
    return l[0];
}

namespace libwalrus {

/**
 * Returns the permanent of a matrix.
 *
 * \rst
 *
 * Returns the permanent of a matrix using Ryser's algorithm 
 * with Gray code ordering, that has a time-complexity of \f$O(n 2^n)\f$
 *
 * \endrst
 *
 * 
 * @tparam T type of the matrix data
 * @param mat a flattened vector of size \f$n^2\f$, representing an
 *      \f$n\times n\f$ row-ordered symmetric matrix.
 * @return permanent of the input matrix
 */
template <typename T>
inline T permanent(std::vector<T> &mat) 
{
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
            llint sgntmp = kg2 - igray(k);
            llint sig = sgntmp/std::abs(sgntmp);
            int pos = n - left_most_set_bit(sgntmp);

            if ( k == threadbound_low[ii] ) {
                chitmp = dec2bin(kg2, n);

                // loop rows of the k-th submatrix
                for ( int j = 0; j < n; j++) {
                    T localsum = static_cast<T>(0);
                    for (int id = 0; id < n; id++) {
                        localsum += static_cast<T>(chitmp[id]) * mat[id*n+j];
                    }
                    tmp[j] += localsum;

                    // update product of row sums 
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
 * Returns the permanent of a matrix using fsum.
 *
 * \rst
 *
 * Returns the permanent of a matrix using Ryser's algorithm 
 * with Gray code ordering.
 *
 * \endrst
 * 
 * 
 * @tparam T type of the matrix data
 * @param mat a flattened vector of size \f$n^2\f$, representing an
 *      \f$n\times n\f$ row-ordered symmetric matrix.
 * @return permanent of the input matrix
 */
template <typename T>
inline double perm_fsum(std::vector<T> &mat) 
{
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
 * Returns the permanent of a matrix using the Ryser's algo 
 * with Gray code ordering
 *
 * \endrst
 *
 *
 * This is a wrapper around the templated function `libwalrus::permanent` 
 * for Python integration. It accepts and returns complex double numeric types, 
 * and returns sensible values for empty and non-even matrices.
 *
 * In addition, this wrapper function automatically casts all matrices
 * to type `complex<long double>`, allowing for greater precision than 
 * supported by Python and NumPy.
 *
 * 
 * @param mat vector representing the flattened matrix
 * @return the permanent
 */
std::complex<double> permanent_quad(std::vector<std::complex<double>> &mat) 
{
    std::vector<std::complex<long double>> matq(mat.begin(), mat.end());
    std::complex<long double> perm = permanent(matq);
    return static_cast<std::complex<double>>(perm);
}

/**
 * \rst
 *
 * Returns the permanent of a matrix using Ryser's algo 
 * with Gray code ordering
 *
 * \endrst
 *
 *
 * This is a wrapper around the templated function `libwalrus::permanent` 
 * for Python integration. It accepts and returns double numeric types, 
 * and returns sensible values for empty and non-even matrices.
 *
 * In addition, this wrapper function automatically casts all matrices
 * to type `long double`, allowing for greater precision than supported
 * by Python and NumPy.
 *
 * 
 * @param mat vector representing the flattened matrix
 * @return the permanent
 */
double permanent_quad(std::vector<double> &mat) 
{
    std::vector<qp> matq(mat.begin(), mat.end());
    qp perm = permanent(matq);
    return static_cast<double>(perm);
}

/**
 * \rst
 *
 * Returns the permanent of a matrix using Ryser's algo 
 * with Gray code ordering with fsum
 *
 * \endrst
 *
 *
 * This is a wrapper around the templated function `libwalrus::perm_fsum` 
 * for Python integration. It accepts and returns double numeric types, 
 * and returns sensible values for empty and non-even matrices.
 *
 *
 * @param mat vector representing the flattened matrix
 * @return the permanent
 */
double permanent_fsum(std::vector<double> &mat) 
{
    std::vector<double> matq(mat.begin(), mat.end());
    double perm = perm_fsum(matq);
    return static_cast<double>(perm);
}


/**
 * Returns the permanent of a matrix (nthreads=1)
 *
 * \rst
 *
 * Returns the permanent of a matrix using the BBFG algorithm.
 * This algorithm was given by Glynn (2010) with the time-complexity 
 * of \f$O(n 2^n)\f$ using Gray code ordering.
 *
 * \endrst
 *
 * 
 * @tparam T type of the matrix data 
 * @param mat a flattened vector of size \f$n^2\f$, representing an
 *      \f$n\times n\f$ row-ordered symmetric matrix.
 * @return permanent of the input matrix
 */
template <typename T>
inline T perm_BBFG_serial(std::vector<T> &mat) 
{
    const size_t n = static_cast<size_t>(std::sqrt(static_cast<double>(mat.size())));
    const double p = pow(2, n-1);
    const ullint x = static_cast<ullint>(p);
    if (p != x) {
        std::cerr << "overflow to inf" << std::endl;
        exit(EXIT_FAILURE);
    }

    constexpr T p1 = static_cast<T>(1.0);
    constexpr T p2 = static_cast<T>(2.0);
    constexpr T n2 = static_cast<T>(-2.0);
    constexpr T zero = static_cast<T>(0);
    
    size_t i, j;
    std::vector<T> colsum(n, zero);
    // init colsum 
    for (i=0; i < n; ++i) {
        for (j=0; j < n; ++j) {
            colsum[i] += mat[j*n + i];
        }
    }

    T mulcolsum, coeff;
    T total = zero;
    ullint k, og=0, ng, gd;
    int sgn=1, gdi;

    // over all 2^{n-1} permutations of delta
    for (k=1; k < x+1; ++k) {
        mulcolsum = std::accumulate(
                        colsum.begin(), 
                        colsum.end(), 
                        p1, 
                        std::multiplies<T>());
        total += sgn > 0 ? mulcolsum : -mulcolsum;
        
        // updating gray order
        ng = igray(k);
        gd = og ^ ng;

        coeff = og > ng ? p2 : n2;
        gdi = left_most_set_bit(gd);
        for (j=0; j < n; ++j) {
            colsum[j] += coeff * mat[gdi * n + j];
        }

        sgn = -sgn;
        og = ng;
    }

    return total / static_cast<T>(x);
}

/**
 * Returns the permanent of a matrix (nthreads=1) (2nd version)
 *
 * \rst
 *
 * Returns the permanent of a matrix using the BBFG algorithm.
 * This algorithm was given by Glynn (2010) with the time-complexity 
 * of \f$O(n 2^n)\f$ using Gray code ordering.
 *
 * \endrst
 * 
 * 
 * @tparam T type of the matrix data
 * @param mat a flattened vector of size \f$n^2\f$, representing an
 *      \f$n\times n\f$ row-ordered symmetric matrix.
 * @return permanent of the input matrix 
 */
template <typename T>
inline T perm_BBFG_serial2(std::vector<T> &mat)
{
    const size_t n = static_cast<size_t>(std::sqrt(static_cast<double>(mat.size())));
    const long double p = pow(2, n-1);
    const T x = static_cast<T>(p);
    if (p == HUGE_VAL || p != x) {
        std::cerr << "overflow to inf" << std::endl;
        exit(EXIT_FAILURE);
    }

    constexpr T p2 = static_cast<T>(2.0);
    constexpr T p1 = static_cast<T>(1.0);
    constexpr T zero = static_cast<T>(0);

    std::vector<size_t> grays(n);
    std::iota(grays.begin(), grays.end(), 0);
    std::vector<T> coeffs(n, p2);
    std::vector<T> colsum(n, zero);
    T mulcolsum, total = p1; 
    size_t i, j, k=0;
    int sgn=1;

    // init colsum 
    for (i=0; i < n; ++i) {
        for (j=0; j < n; ++j) {
            colsum[i] += mat[j*n + i];
        }
        total *= colsum[i];
    }

    while (k < n-1) {
        for (j=0; j < n; ++j) {
            colsum[j] -= coeffs[k] * mat[k*n+j];
        }

        mulcolsum = std::accumulate(
                        colsum.begin(), 
                        colsum.end(), 
                        p1, 
                        std::multiplies<T>());

        coeffs[k] = -coeffs[k];
        sgn = -sgn; 
        total += sgn > 0 ? mulcolsum : -mulcolsum;
        k = next_perm_ordering_index(grays, k);
    }

    return total / x;
}

/**
 * Returns the permanent of a matrix
 *
 * \rst
 *
 * Returns the permanent of a matrix using the BBFG algorithm.
 * This algorithm was given by Glynn (2010) with the time-complexity 
 * of \f$O(n 2^n)\f$ using Gray code ordering.
 *
 * \endrst
 *
 * This is a wrapper function for computing permanent of a matrix 
 * based on Balasubramanian-Bax-Franklin-Glynn (BBFG) formula.
 * 
 * 
 * @tparam T type of the matrix data
 * @param mat a flattened vector of size \f$n^2\f$, representing an
 *      \f$n\times n\f$ row-ordered symmetric matrix.
 * @return permanent of the input matrix
 */
template <typename T>
inline T perm_BBFG(std::vector<T> &mat) {
    return perm_BBFG_serial2(mat);   
}

/**
 * \rst
 *
 * Returns the permanent of a matrix using the BBFG formula.
 * This algorithm was given by Glynn (2010) with the time-complexity 
 * of \f$O(n 2^n)\f$ using Gray code ordering.
 *
 * \endrst
 *
 * This is a wrapper around the templated function `libwalrus::perm_BBFG` 
 * for Python integration. It accepts and returns double numeric types, 
 * and returns sensible values for empty and non-even matrices.
 *
 * In addition, this wrapper function automatically casts all matrices
 * to type `long double`, allowing for greater precision than supported
 * by Python and NumPy.
 *
 * 
 * @param mat vector representing the flattened matrix
 * @return the permanent
 */
double perm_BBFG_qp(std::vector<double> &mat) 
{
    std::vector<qp> matqp(mat.begin(), mat.end());
    qp perm = perm_BBFG(matqp);
    return static_cast<double>(perm);
}

/**
 * \rst
 *
 * Returns the permanent of a matrix using the BBFG formula.
 * This algorithm was given by Glynn (2010) with the time-complexity 
 * of \f$O(n 2^n)\f$ using Gray code ordering.
 *
 * \endrst
 *
 * This is a wrapper around the templated function `libwalrus::perm_BBFG` 
 * for Python integration. It accepts and returns complex double numeric types, 
 * and returns sensible values for empty and non-even matrices.
 *
 * In addition, this wrapper function automatically casts all matrices
 * to type `complex<long double>`, allowing for greater precision than 
 * supported by Python and NumPy.
 *
 * 
 * @param mat vector representing the flattened matrix
 * @return the permanent
 */
std::complex<double> perm_BBFG_cmplx(std::vector<std::complex<double>> &mat) 
{
    std::vector<std::complex<long double>> matx(mat.begin(), mat.end());
    std::complex<long double> perm = perm_BBFG(matx);
    return static_cast<std::complex<double>>(perm);
}

}
