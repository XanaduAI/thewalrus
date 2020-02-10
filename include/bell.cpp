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

//#pragma once
//#include <stdafx.h>
#include <assert.h>
#include <vector>
#include <complex>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <numeric>
#include <complex>
#include <malloc.h>
#include <cstring>

typedef unsigned long long int ullint;


// Single Bell polynomial Z(m,0)
std::complex<double>  Z1(int m, const std::vector<std::complex<double> > &R, 
  const std::vector<std::complex<double> > &y, const std::vector<std::complex<double> > &sqrts,std::vector<std::complex<double> > &cache1) {

    if (m < 0) {
        return 0;
    }

    else if (m == 0) {
        return 1;
    }

    else {
        if (cache1[m].real() == -10) {
            cache1[m] = (Z1(m - 1, R, y, sqrts, cache1) * y[0] + Z1(m - 2, R, y, sqrts, cache1) * R[0] * sqrts[m - 1]) / sqrts[m];
        }
        return cache1[m];
    }
}

// Double Bell polynomial Z(m,n)
double Z2(int m, int n, const std::vector<std::complex<double> > &R, 
  const std::vector<std::complex<double> > &y, const std::vector<std::complex<double> > &sqrts
  ,std::vector<std::complex<double> > &cache1 ,std::vector<std::complex<double> > &cache2) {
    if (n < 0 | m < 0) {
        return 0;
    }

    else if (n == 0) {
        return Z1(m, R, y, sqrts, cache1);
    }

    else {
        int index = dim * m + n;
        if (cache2[m][n].real() == -10) {
            cache2[m][n] = (Z2(m, n - 1) * y[1] +
                            Z2(m - 1, n - 1) * R[1] * sqrts[m] +
                            Z2(m, n - 2) * Q02 * sqrts[n - 1]) / sqrts[n];
        }
        return cache2[m][n];
    }
}





//std::complex<double> *
void hermite_2D(const std::vector<std::complex<double> > &R, const std::vector<std::complex<double> > &y, const int &cutoff) {
//std::complex<double> *hermite_2D() {

    int dim = 2;
    //int cutoff = 10;
//    std::vector<std::complex<double> > R (4, -10);
    //std::vector<std::complex<double> > y (2, -10);
    std::complex<double> Q20 = R[0];
    std::complex<double> Q11 = R[1];
    std::complex<double> Q02 = R[3];
    std::complex<double> Q10 = y[0];
    std::complex<double> Q01 = y[1];

    std::vector<double> sqrts (cutoff, 0);
    for (int k=0; k < dim; k++) {
        sqrts[k]=sqrt(k);
    }
    std::vector <std::complex<double> > cache1 (dim, -10); // set all initial vals to -10
    std::vector <std::vector<std::complex<double> > > cache2 (dim, std::vector<std::complex<double> >(dim, -10));

    ullint Hdim = pow(cutoff, dim);
    std::complex<double> *H;
    H = (std::complex<double>*) malloc(sizeof(std::complex<double>)*Hdim);
    memset(&H[0],0,sizeof(std::complex<double>)*Hdim);






    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            H[cutoff*j+i] = Z2(i, j);
        }
    }


}


int main() {
    return 0;
}