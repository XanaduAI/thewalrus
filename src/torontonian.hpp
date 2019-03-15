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

namespace hafnian {

void find2T (char *dst, Byte len, Byte *pos, char offset)
{
  /* Given a string of length len
       it finds in which positions it has a one
       and stores its position i, as 2*i and 2*i+1 in consecutive slots
       of the array pos.
       It also returns (twice) the number of ones in array dst
  */
  Byte j = offset-1;

  for (Byte i = 0; i<len; i++) {
      if (1 == dst[i]) {
            pos[j] = len-i-1;
            pos[j + offset] = 2*len-i-1;
            j--;
        }
    }
}

char sum(char *dst, Byte m){
  char sum_tot = 0;
  for(int i=0;i<m;i++) {
    sum_tot += (Byte)dst[i];
  }
  return sum_tot;
}


template <typename T>
inline long double torontonian(std::vector<T> &mat) {
    // Here weinput the matrix from python. The variable n is the size of the matrix
    int n = std::sqrt(static_cast<double>(mat.size()));
    Byte m = n/2;
    unsigned long long int x = static_cast<unsigned long long int>(pow(2,m));

    long double netsum = 0;

    namespace eg = Eigen;
    eg::Matrix<T,eg::Dynamic,eg::Dynamic> A = eg::Map<eg::Matrix<T,eg::Dynamic,eg::Dynamic>, eg::Unaligned>(mat.data(), n, n);

    for (int k = 0; k < x; k++){
        unsigned long long int xx = k;
        char* dst = new char[m];

        dec2bin(dst,xx,m);
        char len = sum(dst,m);

        Byte* short_st = new Byte[2*len];
        find2T(dst, m, short_st, len);
        delete [] dst;

        // eg::Matrix<T,eg::Dynamic,eg::Dynamic> B(2*len, 2*len, 0.);
        eg::Matrix<T,eg::Dynamic,eg::Dynamic> B;
        B.resize(2*len, 2*len);

        for (int i = 0; i < 2*len; i++){
            for (int j = 0; j < 2*len; j++){
                B(i, j) = -A(short_st[i], short_st[j]);
            }
        }

        delete [] short_st;

        for (int i = 0; i < 2*len; i++){
            B(i, i) += 1;
        }

        long double det = std::real(B.determinant());

        if(len % 2 ==0){
            netsum += 1.0/std::sqrt(det);
        }
        else{
            netsum -= 1.0/std::sqrt(det);
        }
        // The set of integers that we will use to generate the new matrix is in the array short_st which has length 2*len
        // Then we calculate the det of the subarray
        // we add it the sign (-1)^len and we are done
    }
    return netsum;
}


template <typename T>
inline double torontonian_fsum(std::vector<T> &mat) {
    // Here weinput the matrix from python. The variable n is the size of the matrix
    int n = std::sqrt(static_cast<double>(mat.size()));
    Byte m = n/2;
    unsigned long long int x = static_cast<unsigned long long int>(pow(2,m));

    fsum::sc_partials netsum;

    namespace eg = Eigen;
    eg::Matrix<T,eg::Dynamic,eg::Dynamic> A = eg::Map<eg::Matrix<T,eg::Dynamic,eg::Dynamic>, eg::Unaligned>(mat.data(), n, n);

    for (int k = 0; k < x; k++){
        unsigned long long int xx = k;
        char* dst = new char[m];

        dec2bin(dst,xx,m);
        char len = sum(dst,m);

        Byte* short_st = new Byte[2*len];
        find2T(dst, m, short_st, len);
        delete [] dst;

        // eg::Matrix<double,eg::Dynamic,eg::Dynamic> B(2*len, 2*len, 0.);
        eg::Matrix<T,eg::Dynamic,eg::Dynamic> B;
        B.resize(2*len, 2*len);

        for (int i = 0; i < 2*len; i++){
            for (int j = 0; j < 2*len; j++){
                B(i, j) = -A(short_st[i], short_st[j]);
            }
        }

        delete [] short_st;

        for (int i = 0; i < 2*len; i++){
            B(i, i) += 1;
        }

        long double det = std::real(B.determinant());

        if(len % 2 ==0){
            netsum += 1.0/std::sqrt(det);
        }
        else{
            netsum += -1.0/std::sqrt(det);
        }
        // The set of integers that we will use to generate the new matrix is in the array short_st which has length 2*len
        // Then we calculate the det of the subarray
        // we add it the sign (-1)^len and we are done
    }
    return static_cast<double>(netsum);
}

double torontonian_quad(std::vector<std::complex<double>> &mat) {
    std::vector<std::complex<long double>> matq(mat.begin(), mat.end());
    long double tor = torontonian(matq);
    return static_cast<double>(tor);
}

double torontonian_quad(std::vector<double> &mat) {
    std::vector<long double> matq(mat.begin(), mat.end());
    long double tor = torontonian(matq);
    return static_cast<double>(tor);
}

}
