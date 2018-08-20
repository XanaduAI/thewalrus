// Copyright 2018 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lhafnian.h"
using namespace std;

/*
    Given a complex matrix z of dimensions n x n it calculates traces[k] = tr(z^j) for 1 <= j <= l
*/
void powtrace(CplxType* z, int n, CplxType* traces, int l) {
    CplxType* vals = new CplxType[n];
    CplxType* pvals = new CplxType[n];

    // work arrays
    int lwork = 2 * n;
    CplxType* work = new CplxType[lwork];
    double* rwork = new double[n];

    if (n != 0) {
        //evals(z, vals, n, work, lwork, rwork); // TODO: Uncomment and link to actual method?
    }

    for (int j = 0; j < n; j++) {
        pvals[j] = vals[j];
    }
    for (int i = 0; i < l; i++) {
        CplxType sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += pvals[j];
        }
        traces[i] = sum;
        for (int j = 0; j < n; j++) {
            pvals[j] = pvals[j] * vals[j];
        }
    }

    delete[] vals, pvals, work, rwork;
}

/*
    Convert decimal number in x to character vector dst of length len representing binary number
*/
void dec2bin(char* dst, unsigned long long int x, Byte len) {
    char i; // this variable cannot be unsigned
    for (i = len - 1; i >= 0; i--) {    
        *dst++ = x >> i & 1;
    }
}

/*
    Given a string of length len it finds in which positions it has a one and stores its position i, 
    as 2*i and 2*i+1 in consecutive slots of the array pos.
    It also returns (twice) the number of ones in array dst
*/
Byte find2(char* dst, Byte len, Byte* pos) {
    Byte j = 0;
    for (Byte i = 0; i < len; i++) {
        if (1 != dst[i])
            break;
        pos[2 * j] = 2 * i;
        pos[2 * j + 1] = 2 * i + 1;
        j++;
    }
    return 2 * j;
}

// Functions for computing Hafnians without considering loops (diagonal elements).

/*
    This function calculates adds parts X to X+chunksize of Cygan and Pilipczuk formula for the Hafnian of matrix mat
*/
CplxType do_chunk(CplxType* mat, int n, unsigned long long int X, unsigned long long int chunksize) {
    CplxType res = 0.0;
    Byte m = n / 2;
    
    #pragma omp parallel for
    for (unsigned long long int x = X; x < X + chunksize; x++) {
        CplxType summand = 0.0;
        CplxType factor;
        CplxType powfactor;

        char* dst = new char[m];
        Byte* pos = new Byte[n];
        dec2bin(dst, x, m);

        char sum = find2(dst, m, pos);
        CplxType* B = new CplxType[sum * sum];
        for (int i = 0; i < sum; i++)     {
            for (int j = 0; j < sum; j++) {
                B[i * sum + j] = (CplxType) mat[pos[i] * n + ((pos[j]) ^ 1)];
            }
        }
        
        CplxType* traces = new CplxType[m];
        powtrace(B, sum, traces, m);
        
        char cnt;
        Byte cntindex;
        cnt = 1;
        cntindex = 0;
        
        CplxType** comb = new CplxType*[2];
        for (int i = 0; i < 2; i++) {
            comb[i] = new CplxType[m + 1];
            for (int j = 0; j < m + 1; j++) {
                comb[i][j] = 0;
            }
        }
        comb[0][0] = 1.0;

        for (int i = 1; i <= n / 2; i++) {
            factor = traces[i - 1] / ((double) 2 * i);
            powfactor = 1.0;
            //the following line is significantly different than in Octave
            //It basically reassings the first column of comb to the second and viceversa
            cnt = -cnt;
            cntindex = (1 + cnt) / 2;
            for (int j = 0; j < n / 2 + 1; j++) {
                comb[1 - cntindex][j] = comb[cntindex][j];
            }
            for (int j = 1; j <= (n / (2 * i)); j++) {
                powfactor = powfactor * factor / (double) j;
                for (int k = i * j + 1; k <= n / 2 + 1; k++) {
                    comb[1 - cntindex][k - 1] += comb[cntindex][k - i * j - 1] * powfactor;
                }
            }
        }
        if (((sum / 2) % 2) == (n / 2 % 2)) {
            summand = comb[1 - cntindex][n / 2];
        } else {
            summand = -comb[1 - cntindex][n / 2];
        }

        #pragma omp critical // TODO: There's better OpenMP constructs for handling this...
        res += summand;

        // TODO: Can probably optimise array allocation to be outside the loop -- dimensions are independent of x
        delete[] comb[0], comb[1];
        delete[] dst, pos, B, traces, comb;
    }

    return res;
}

/*
    This is a wrapper for the function hafnian. Instead of returning the Hafnian of the matrix mat by value 
    it does by reference with two doubles for the real an imaginary respectively.
*/
void haf(CplxType* mat, int n, double* res) {
    CplxType result = hafnian(mat, n);
    res[0] = real(result);
    res[1] = imag(result);
}

void dhaf(double* mat, int n, double *res) {
    haf((CplxType*) mat, n, res); // XXX: This looks pretty sketchy -- it doesn't IOOBE in C99?? Presumably does in C++...
}

/*
    Add all the terms necessary to calculate the Hafnian of the matrix mat of size n
    It additionally uses OpenMP to parallelize the summation
*/
CplxType hafnian(CplxType* mat, int n) {
    assert(n % 2 == 0);
    Byte m = n / 2;
    Byte mm = m / 2;

    unsigned long long int pow1 = ((unsigned long long int)pow(2.0, (double)m)); //-1;
    unsigned long long int workers = ((unsigned long long int)pow(2.0, (double)mm));
    workers = min(workers, pow1);

    unsigned long long int chunksize = pow1;
    unsigned long long int rank = 0;

    return do_chunk(mat, n, rank, chunksize);
}

CplxType dhafnian(double *mat, int n) {
    return hafnian((CplxType*)mat, n); // XXX: Same sketchiness...
}

// Functions for computing Hafnians WITH considering loops (diagonal elements). These functions are named with a suffix "_loops" to differentiate them from previous ones.

/*
    This function calculates adds parts X to X+chunksize of Cygan and Pilipczuk formula for the Hafnian of matrix mat
*/
CplxType do_chunk_loops(CplxType* mat, CplxType* C, CplxType* D, int n, unsigned long long int X, unsigned long long int chunksize) {
    CplxType res = 0.0;

    #pragma omp parallel for
    for (unsigned long long int x = X * chunksize; x < (X + 1) * chunksize; x++) {
        CplxType summand = 0.0;
        Byte m = n / 2;
        CplxType factor, powfactor;

        char* dst = new char[m];
        Byte* pos = new Byte[n];
        dec2bin(dst, x, m);
        Byte sum = find2(dst, m, pos);

        CplxType* B = new CplxType[sum * sum];
        CplxType* B_powtrace = new CplxType[sum * sum];
        CplxType* C1 = new CplxType[sum];
        CplxType* D1 = new CplxType[sum];

        for (int i = 0; i < sum; i++) {
            for (int j = 0; j < sum; j++) {
                B[i * sum + j] = mat[pos[i] * n + ((pos[j]) ^ 1)];
                B_powtrace[i * sum + j] = mat[pos[i] * n + ((pos[j]) ^ 1)];
            }
            C1[i] = C[pos[i]];
            D1[i] = D[pos[i]];
        }

        CplxType* traces = new CplxType[m];
        powtrace(B_powtrace, sum, traces, m);
        char cnt = 1;
        Byte cntindex = 0;
        CplxType** comb = new CplxType*[2];
        for (int i = 0; i < 2; i++) {
            comb[i] = new CplxType[m + 1];
            for (int j = 0; j < m + 1; j++) {
                comb[i][j] = 0;
            }
        }
        comb[0][0] = 1.0;

        for (int i = 1; i <= n / 2; i++) {
            factor = traces[i - 1] / (2. * i);

            CplxType tmpn = 0.0;
            for (int k = 0; k < sum; k++) {
                tmpn += C1[k] * D1[k];
            }
            factor += 0.5 * tmpn;

            CplxType* tmp_c1 = new CplxType[sum];
            CplxType tmp = 0.0;
            for (int k = 0; k < sum; k++) {
                tmp = 0.0;
                for (int j = 0; j < sum; j++) {
                    tmp += C1[j] * B[j * sum + k];
                }
                tmp_c1[k] = tmp;
                C1[k] = tmp_c1[k];
            }

            powfactor = 1.0;
            //the following line is significantly different than in Octave
            //It basically reassigns  the first column of comb to the second and viceversa
            cnt = -cnt;
            cntindex = (1 + cnt) / 2;
            for (int j = 0; j < n / 2 + 1; j++) {
                comb[1 - cntindex][j] = comb[cntindex][j];
            }

            for (int j = 1; j <= (n / (2 * i)); j++) {
                powfactor = powfactor * factor / (CplxType) j;
                for (int k = i * j + 1; k <= n / 2 + 1; k++) {
                    comb[1 - cntindex][k - 1] = comb[1 - cntindex][k - 1] + comb[cntindex][k - i * j - 1] * powfactor;
                }
            }
            
            delete[] tmp_c1;
        }

        if (((sum / 2) % 2) == (n / 2 % 2)) {
            summand = comb[1 - cntindex][n / 2];
        } else {
            summand = -comb[1 - cntindex][n / 2];
        }
        #pragma omp critical
        res += summand;

        // TODO: More potential array allocation optimisations...
        delete[] comb[0], comb[1];
        delete[] dst, pos, B, B_powtrace, C1, D1, traces, comb;
    }

    return res;
}

/*
    Add all the terms necessary to calculate the Hafnian of the matrix mat of size n
    It additionally uses open MP to parallelize the summation
*/
CplxType hafnian_loops(CplxType* mat, int n) {
    assert(n % 2 == 0);
    Byte m = n / 2;
    Byte mm = m / 2;
    unsigned long long int pow1 = ((unsigned long long int)pow(2.0, (double)m)); //-1;
    unsigned long long int workers = ((unsigned long long int)pow(2.0, (double)mm));
    workers = min(workers, pow1);

    unsigned long long int chunksize = pow1;
    unsigned long long int rank = 0;

    // CplxType D_mat[n][n] ;
    CplxType* C = new CplxType[n];
    CplxType* D = new CplxType[n];
    for (int i = 0; i < n; i++) {
        C[i] = 0.0;
        D[i] = 0.0;
    }

    for (int i = 0; i < n; i++)
    {
        // D_mat[i][i] = mat[i * n + i];
        D[i] = mat[i * n + i];
    }

    for (int i = 0; i < n; i += 2) {
        C[i] = D[i + 1];
        C[i + 1] = D[i];
    }

    CplxType summand = do_chunk_loops(mat, C, D, n, rank, chunksize);

    delete[] C, D;

    return summand;
}

CplxType dhafnian_loops(double *mat, int n) {
    return hafnian_loops((CplxType*) mat, n); // XXX: Yet more sketchiness...
}

/*
    This is a wrapper for the function hafnian. Instead of returning the Hafnian of the matrix
    mat by value it does by reference with two doubles for the real an imaginary respectively.
*/
void haf_loops(CplxType* mat, int n, double* res) {
    CplxType result = hafnian_loops(mat, n);
    res[0] = result.real();
    res[1] = result.imag();
}

void dhaf_loops(double* mat, int n, double* res) {
    haf_loops((CplxType*) mat, n, res); // XXX: Viva la sketchiness...
}