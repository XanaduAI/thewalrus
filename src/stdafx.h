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
#include <iomanip>
#include <iostream>
#include <vector>
#include <numeric>
#include <complex>
#include <assert.h>

#define SC_STACK  128        // 2098 bit / 53 = min 40 long doubles

#ifdef _OPENMP
#include <omp.h>
#endif

typedef unsigned char Byte;
typedef std::complex<double> double_complex;
typedef std::vector<double_complex> vec_complex;
typedef std::vector<double> vec_double;

/**
 * Convert decimal number `x` to character vector `dst` of length `len`
 * representing a binary number
 *
 * @param dst character array to store the binary number
 * @param x decimal integer
 * @param len length of the binary character array
 */
inline void dec2bin(char* dst, unsigned long long int x, Byte len) {
    char i; // this variable cannot be unsigned
    for (i = len - 1; i >= 0; --i) {
        *dst++ = x >> i & 1;
    }
}


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
inline Byte find2(char* dst, Byte len, Byte* pos) {
    Byte j = 0;
    for (Byte i = 0; i < len; i++) {
        if (1 == dst[i]) {
            pos[2 * j] = 2 * i;
            pos[2 * j + 1] = 2 * i + 1;
            j++;
        }
    }
    return 2 * j;
}
