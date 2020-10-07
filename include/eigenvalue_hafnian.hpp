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
 * Contains functions for computing the hafnian using the algorithm described in
 * *A faster hafnian formula for complex matrices and its benchmarking
 * on the Titan supercomputer*, [arxiv:1805.12498](https://arxiv.org/abs/1805.12498)
 */


#include "stdafx.h"
#include "powtrace.hpp"

namespace libwalrus {

/**
 * Calculates the partial sum \f$X,X+1,\dots,X+\text{chunksize}\f$ using
 * the Cygan and Pilipczuk formula for the hafnian of matrix `mat`.
 *
 * Note that if `X=0` and `chunksize=pow(2.0, n/2)`, then the full hafnian is calculated.
 *
 * This function uses OpenMP (if available) to parallelize the reduction.
 *
 * @param mat vector representing the flattened matrix
 * @param n size of the matrix
 * @param X initial index of the partial sum
 * @param chunksize length of the partial sum
 * @return the partial sum for hafnian
 */
template <typename T>
inline T do_chunk(std::vector<T> &mat, int n, unsigned long long int X, unsigned long long int chunksize) {
    // This function calculates adds parts X to X+chunksize of Cygan and Pilipczuk formula for the
    // Hafnian of matrix mat

    T res = 0.0;

    #pragma omp parallel for
    for (unsigned long long int x = X; x < X + chunksize; x++) {
        T summand = 0.0;

        Byte m = n / 2;
        int i, j, k;
        T factor, powfactor;

        char* dst = new char[m];
        Byte* pos = new Byte[n];
        dec2bin(dst, x, m);
        Byte sum = find2(dst, m, pos);
        delete [] dst;

        std::vector<T> B(sum * sum, 0.0);

        for (i = 0; i < sum; i++) {
            for (j = 0; j < sum; j++) {
                B[i * sum + j] = mat[pos[i] * n + ((pos[j]) ^ 1)];
            }
        }
        delete [] pos;

        std::vector<T> traces(m, 0.0);
        if (sum != 0) {
            traces = powtrace(B, sum, m);
        }

        char cnt = 1;
        Byte cntindex = 0;

        std::vector<T> comb(2 * (m + 1), 0.0);
        comb[0] = 1.0;

        for (i = 1; i <= n / 2; i++) {
            factor = traces[i - 1] / static_cast<T>(2.0 * i);
            powfactor = 1.0;

            cnt = -cnt;
            cntindex = (1 + cnt) / 2;
            for (j = 0; j < n / 2 + 1; j++) {
                comb[(m + 1) * (1 - cntindex) + j] = comb[(m + 1) * cntindex + j];
            }
            for (j = 1; j <= (n / (2 * i)); j++) {
                powfactor = powfactor * factor / (1.0 * j);
                for (k = i * j + 1; k <= n / 2 + 1; k++) {
                    comb[(m + 1) * (1 - cntindex) + k - 1] += comb[(m + 1) * cntindex + k - i * j - 1] * powfactor;
                }
            }
        }
        if (((sum / 2) % 2) == (n / 2 % 2)) {
            summand = comb[(m + 1) * (1 - cntindex) + n / 2];
        }
        else {
            summand = -comb[(m + 1) * (1 - cntindex) + n / 2];
        }
        #pragma omp critical
        res += summand;
    }

    return res;
}

/**
 * Calculates the partial sum \f$X,X+1,\dots,X+\text{chunksize}\f$ using
 * the Cygan and Pilipczuk formula for the loop hafnian of matrix `mat`.
 *
 * Note that if `X=0` and `chunksize=pow(2.0, n/2)`, then the full loop hafnian is calculated.
 *
 * This function uses OpenMP (if available) to parallelize the reduction.
 *
 * @param mat vector representing the flattened matrix
 * @param C contains the diagonal elements of matrix ``z``
 * @param D the diagonal elements of matrix ``z``, with every consecutive pair
 *      swapped (i.e., ``C[0]==D[1]``, ``C[1]==D[0]``, ``C[2]==D[3]``,
 *      ``C[3]==D[2]``, etc.).
 * @param n size of the matrix
 * @param X initial index of the partial sum
 * @param chunksize length of the partial sum
 * @return the partial sum for the loop hafnian
 */
template <typename T>
inline T do_chunk_loops(std::vector<T> &mat, std::vector<T> &C, std::vector<T> &D, int n, unsigned long long int X, unsigned long long int chunksize) {

    T res = 0.0;

    #pragma omp parallel for
    for (unsigned long long int x = X * chunksize; x < (X + 1)*chunksize; x++) {
        T summand = 0.0;

        Byte m = n / 2;
        int i, j, k;
        T factor, powfactor;

        char* dst = new char[m];
        Byte* pos = new Byte[n];
        dec2bin(dst, x, m);
        Byte sum = find2(dst, m, pos);
        delete [] dst;

        std::vector<T> B(sum * sum, 0.0), B_powtrace(sum * sum, 0.0);
        std::vector<T> C1(sum, 0.0), D1(sum, 0.0);

        for (i = 0; i < sum; i++) {
            for (j = 0; j < sum; j++) {
                B[i * sum + j] = mat[pos[i] * n + ((pos[j]) ^ 1)];
                B_powtrace[i * sum + j] = mat[pos[i] * n + ((pos[j]) ^ 1)];
            }
            C1[i] = C[pos[i]];
            D1[i] = D[pos[i]];
        }
        delete [] pos;

        std::vector<T> traces(m, 0.0);
        if (sum != 0) {
            traces = powtrace(B, sum, m);
        }

        char cnt = 1;
        Byte cntindex = 0;

        std::vector<T> comb(2 * (m + 1), 0.0);
        comb[0] = 1.0;

        for (i = 1; i <= n / 2; i++) {
            factor = traces[i - 1] / static_cast<T> (2.0 * i);
            T tmpn = 0.0;

            for (int i = 0; i < sum; i++) {
                tmpn += C1[i] * D1[i];
            }

            factor += static_cast<T> (0.5) * tmpn;
            std::vector<T> tmp_c1(sum, 0.0);

            T tmp = 0.0;

            for (int i = 0; i < sum; i++) {
                tmp = 0.0;

                for (int j = 0; j < sum; j++) {
                    tmp += C1[j] * B[j * sum + i];
                }

                tmp_c1[i] = tmp;
            }

            for (int i = 0; i < sum; i++) {
                C1[i] = tmp_c1[i];
            }

            powfactor = 1.0;

            cnt = -cnt;
            cntindex = (1 + cnt) / 2;
            for (j = 0; j < n / 2 + 1; j++) {
                comb[(m + 1) * (1 - cntindex) + j] = comb[(m + 1) * cntindex + j];
            }

            for (j = 1; j <= (n / (2 * i)); j++) {
                powfactor = powfactor * factor / static_cast<T> (1.0 * j);
                for (k = i * j + 1; k <= n / 2 + 1; k++) {
                    comb[(m + 1) * (1 - cntindex) + k - 1] = comb[(m + 1) * (1 - cntindex) + k - 1] + comb[(m + 1) * cntindex + k - i * j - 1] * powfactor;
                }
            }
        }

        if (((sum / 2) % 2) == (n / 2 % 2)) {
            summand = comb[(m + 1) * (1 - cntindex) + n / 2];
        }
        else {
            summand = -comb[(m + 1) * (1 - cntindex) + n / 2];
        }
        #pragma omp critical
        res += summand;
    }

    return res;
}


/**
* Returns the hafnian of a matrix using the algorithm described in
* *A faster hafnian formula for complex matrices and its benchmarking
* on the Titan supercomputer*, [arxiv:1805.12498](https://arxiv.org/abs/1805.12498>).
*
* @param mat a flattened vector of size \f$n^2\f$, representing an \f$n\times n\f$
*       row-ordered symmetric matrix.
* @return hafnian of the input matrix
*/
template <typename T>
inline T hafnian(std::vector<T> &mat) {
    int n = std::sqrt(static_cast<double>(mat.size()));
    assert(n % 2 == 0);

    Byte m = n / 2;
    Byte mm = m / 2;
    unsigned long long int pow1 = ((unsigned long long int)pow(2.0, (double)m));
    unsigned long long int workers = ((unsigned long long int)pow(2.0, (double)mm));
    workers = std::min(workers, pow1);

    unsigned long long int chunksize = pow1;
    unsigned long long int rank = 0;

    T haf;
    haf = do_chunk(mat, n, rank, chunksize);
    return  haf;
}


/**
* Returns the loop hafnian of a matrix using the algorithm described in
* *A faster hafnian formula for complex matrices and its benchmarking
* on the Titan supercomputer*, [arxiv:1805.12498](https://arxiv.org/abs/1805.12498>).
*
* @param mat a flattened vector of size \f$n^2\f$, representing an \f$n\times n\f$
*       row-ordered symmetric matrix.
* @return hafnian of the input matrix
*/
template <typename T>
inline T loop_hafnian(std::vector<T> &mat) {
    int n = std::sqrt(static_cast<double>(mat.size()));
    assert(n % 2 == 0);

    Byte m = n / 2;
    Byte mm = m / 2;
    unsigned long long int pow1 = ((unsigned long long int)pow(2.0, (double)m));
    unsigned long long int workers = ((unsigned long long int)pow(2.0, (double)mm));
    workers = std::min(workers, pow1);

    std::vector<T> D(n, 0.0), C(n, 0.0);

    unsigned long long int chunksize = pow1;
    unsigned long long int rank = 0;

    for (int i = 0; i < n; i++) {
        D[i] = mat[i * n + i];
    }

    for (int i = 0; i < n; i += 2) {
        C[i] = D[i + 1];
        C[i + 1] = D[i];
    }

    T haf;
    haf = do_chunk_loops(mat, C, D, n, rank, chunksize);
    return  haf;
}


/**
* Returns the hafnian of a matrix using the algorithm described in
* *A faster hafnian formula for complex matrices and its benchmarking
* on the Titan supercomputer*, [arxiv:1805.12498](https://arxiv.org/abs/1805.12498>).
*
* This is a wrapper around the templated function hafnian() for Python
* integration. It accepts and returns complex double numeric types, and
* returns sensible values for empty and non-even matrices.
*
* @param mat a flattened vector of size \f$n^2\f$, representing an \f$n\times n\f$
*       row-ordered symmetric matrix.
* @return hafnian of the input matrix
*/
std::complex<double> hafnian_eigen(std::vector<std::complex<double>> &mat) {
    std::vector<std::complex<double>> matq(mat.begin(), mat.end());
    int n = std::sqrt(static_cast<double>(mat.size()));
    std::complex<double> haf;

    if (n == 0)
        haf = std::complex<double>(1.0, 0.0);
    else if (n % 2 != 0)
        haf = std::complex<double>(0.0, 0.0);
    else
        haf = hafnian(matq);

    return haf;
}


/**
* Returns the hafnian of a matrix using the algorithm described in
* *A faster hafnian formula for complex matrices and its benchmarking
* on the Titan supercomputer*, [arxiv:1805.12498](https://arxiv.org/abs/1805.12498>).
*
* This is a wrapper around the templated function hafnian() for Python
* integration. It accepts and returns double numeric types, and
* returns sensible values for empty and non-even matrices.
*
* @param mat a flattened vector of size \f$n^2\f$, representing an \f$n\times n\f$
*       row-ordered symmetric matrix.
* @return hafnian of the input matrix
*/
double hafnian_eigen(std::vector<double> &mat) {
    std::vector<double> matq(mat.begin(), mat.end());
    int n = std::sqrt(static_cast<double>(mat.size()));
    double haf;

    if (n == 0)
        haf = 1.0;
    else if (n % 2 != 0)
        haf = 0.0;
    else
        haf = static_cast<double>(hafnian(matq));

    return haf;
}


/**
* Returns the loop hafnian of a matrix using the algorithm described in
* *A faster hafnian formula for complex matrices and its benchmarking
* on the Titan supercomputer*, [arxiv:1805.12498](https://arxiv.org/abs/1805.12498>).
*
* This is a wrapper around the templated function libwalrus::loop_hafnian() for Python
* integration. It accepts and returns complex double numeric types, and
* returns sensible values for empty and non-even matrices.
*
* @param mat a flattened vector of size \f$n^2\f$, representing an \f$n\times n\f$
*       row-ordered symmetric matrix.
* @return loop hafnian of the input matrix
*/
std::complex<double> loop_hafnian_eigen(std::vector<std::complex<double>> &mat) {
    std::vector<std::complex<double>> matq(mat.begin(), mat.end());
    int n = std::sqrt(static_cast<double>(mat.size()));
    std::complex<double> haf;

    if (n == 0)
        haf = std::complex<double>(1.0, 0.0);
    else if (n % 2 != 0) {
    	std::vector<std::complex<double>> matq2((n + 1) * (n + 1), std::complex<double>(0.0, 0.0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matq2[i * (n + 1) + j] = matq[i * n + j];
            }
        }
        matq2[(n + 1) * (n + 1) - 1] = std::complex<double>(1.0, 0.0);


        haf = loop_hafnian(matq2);
    }
    else
        haf = loop_hafnian(matq);

    return haf;
}


/**
* Returns the loop hafnian of a matrix using the algorithm described in
* *A faster hafnian formula for complex matrices and its benchmarking
* on the Titan supercomputer*, [arxiv:1805.12498](https://arxiv.org/abs/1805.12498>).
*
* This is a wrapper around the templated function loop_hafnian() for Python
* integration. It accepts and returns double numeric types, and
* returns sensible values for empty and non-even matrices.
*
* @param mat a flattened vector of size \f$n^2\f$, representing an \f$n\times n\f$
*       row-ordered symmetric matrix.
* @return loop hafnian of the input matrix
*/
double loop_hafnian_eigen(std::vector<double> &mat) {
    std::vector<double> matq(mat.begin(), mat.end());
    int n = std::sqrt(static_cast<double>(mat.size()));
    double haf;

    if (n == 0)
        haf = 1.0;
    else if (n % 2 != 0) {
    	std::vector<double> matq2((n + 1) * (n + 1), 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matq2[i * (n + 1) + j] = mat[i * n + j];
            }
        }
        matq2[(n + 1) * (n + 1) - 1] = 1.0;


        haf = loop_hafnian(matq2);
    }
    else
        haf = static_cast<double>(loop_hafnian(matq));

    return haf;
}

/**
 * \rst
 *
 * Returns the loop hafnian of a matrix using the trace algorithm described in
 * *A faster hafnian formula and its benchmarking in a supercomputer* :cite:`bjorklund2019faster`,
 *
 * \endrst
 *
 * This is a wrapper around the templated function `libwalrus::loop_hafnian` for Python
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

std::complex<double> loop_hafnian_quad(std::vector<std::complex<double>> &mat) {
    std::vector<std::complex<long double>> matq(mat.begin(), mat.end());
    int n = std::sqrt(static_cast<double>(mat.size()));
    std::complex<long double> haf;

    if (n == 0)
        haf = std::complex<double>(1.0, 0.0);
    else if (n % 2 != 0) {
    	std::vector<std::complex<long double>> matq2((n + 1) * (n + 1), std::complex<long double>(0.0, 0.0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matq2[i * (n + 1) + j] = matq[i * n + j];
            }
        }
        matq2[(n + 1) * (n + 1) - 1] = std::complex<long double>(1.0, 0.0);


        haf = loop_hafnian(matq2);
    }
    else
        haf = loop_hafnian(matq);

    return static_cast<std::complex<double>>(haf);
}

/**
 * \rst
 *
 * Returns the loop hafnian of a matrix using the trace algorithm described in
 * *A faster hafnian formula and its benchmarking in a supercomputer* :cite:`bjorklund2019faster`,
 *
 * \endrst
 *
 * This is a wrapper around the templated function `libwalrus::loop_hafnian` for Python
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

double loop_hafnian_quad(std::vector<double> &mat) {
    std::vector<long double> matq(mat.begin(), mat.end());
    int n = std::sqrt(static_cast<double>(mat.size()));
    long double haf;

    if (n == 0)
        haf = 1.0;
    else if (n % 2 != 0) {
    	std::vector<long double> matq2((n + 1) * (n + 1), 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matq2[i * (n + 1) + j] = mat[i * n + j];
            }
        }
        matq2[(n + 1) * (n + 1) - 1] = 1.0;


        haf = loop_hafnian(matq2);
    }
    else
        haf = loop_hafnian(matq);

    return static_cast<double>(haf);
}

}
