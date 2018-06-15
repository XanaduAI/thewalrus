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
#include <stdio.h>


void
powtrace(double complex z[], int n, telem traces[], int l)
{
    /*
      given a complex matrix z of dimensions n x n
      it calculates traces[k] = tr(z^j) for 1 <= j <= l
     */
    double complex vals[n];
    telem pvals[n];

    // work arrays
    int lwork = 2*n;
    double complex work[lwork];
    double rwork[n];

    if (n != 0){
        evals(z, vals, n, work, lwork, rwork);
    }

    int i,j;
    telem sum;
    for(j = 0; j < n; j++)
    {
        pvals[j] = vals[j];
    }
    for(i = 0; i < l; i++)
    {
        sum = 0.0;
        for(j = 0; j < n; j++)
        {
            sum += pvals[j];
        }
        traces[i] = sum;
        for(j = 0; j < n; j++)
        {
            pvals[j] = pvals[j] * vals[j];
        }
    }
}

void
dec2bin(char *dst, unsigned long long int x, sint len)
{
    /*
    Convert decimal number in  x to character vector dst of length len
    representing binary number
    */

    char i; // this variable cannot be unsigned
    for (i = len - 1; i >= 0; --i)
        *dst++ = x >> i & 1;
}


unsigned char
find2(char *dst, sint len, sint *pos)
{
    /* Given a string of length len
       it finds in which positions it has a one
       and stores its position i, as 2*i and 2*i+1 in consecutive slots
       of the array pos.
       It also returns (twice) the number of ones in array dst
    */
    unsigned char sum = 0;
    unsigned char j = 0;
    for(unsigned char i = 0; i < len; i++)
    {
        if(1 == dst[i])
        {
            sum++;
            pos[2 * j] = 2 * i;
            pos[2 * j + 1] = 2 * i + 1;
            j++;
        }
    }
    return 2 * sum;
}

// Functions for computing Hafnians without considering loops (diagonal elements).

telem
do_chunk (telem mat[], int n, unsigned long long int X,
          unsigned long long int chunksize)
{
    /*
       This function calculates adds parts X to X+chunksize of Cygan and Pilipczuk formula for the
       Hafnian of matrix mat
     */

    telem res = 0.0;
    #pragma omp parallel
    {
        #pragma omp for
        for (unsigned long long int x = X; x < X + chunksize; x++)
        {

            telem summand = 0.0;
            sint m = n / 2;
            int i, j, k;
            telem factor;
            telem powfactor;


            char dst[m];
            sint pos[n];
            char sum;
            dec2bin (dst, x, m);
            sum = find2 (dst, m, pos);
            double complex B[sum * sum];
            for (i = 0; i < sum; i++)
            {
                for (j = 0; j < sum; j++)
                {
                    B[i * sum + j] =
                        (double complex) mat[pos[i] * n + ((pos[j]) ^ 1)];
                }
            }
            telem traces[m];
            powtrace (B, sum, traces, m);
            char cnt;
            sint cntindex;
            cnt = 1;
            cntindex = 0;
            telem comb[2][m + 1];
            for (i = 0; i < n / 2 + 1; i++)
            {
                comb[0][i] = 0.0;
            }
            comb[0][0] = 1.0;

            for (i = 1; i <= n / 2; i++)
            {
                factor = traces[i - 1] / (2 * i);
                powfactor = 1.0;
                //the following line is significantly different than in Octave
                //It basically reassings  the first column of comb to the second and viceversa
                cnt = -cnt;
                cntindex = (1 + cnt) / 2;
                for (j = 0; j < n / 2 + 1; j++)
                {
                    comb[1 - cntindex][j] = comb[cntindex][j];
                }
                for (j = 1; j <= (n / (2 * i)); j++)
                {
                    powfactor = powfactor * factor / j;
                    for (k = i * j + 1; k <= n / 2 + 1; k++)
                    {
                        comb[1 - cntindex][k - 1] +=
                            comb[cntindex][k - i * j - 1] * powfactor;
                    }
                }
            }
            if (((sum / 2) % 2) == (n / 2 % 2))
            {
                summand = comb[1 - cntindex][n / 2];
            }
            else
            {
                summand = -comb[1 - cntindex][n / 2];
            }
            #pragma omp critical
            res += summand;
        }
    }

    return res;
}


void
haf(telem mat[], int n, double res[])
{
    /*
      This is a wrapper for the function hafnian. Instead of returning the Hafnian of the matrix
      mat by value it does by reference with two doubles for the real an imaginary respectively.
    */

    telem result = hafnian(mat, n);
    res[0] = creal(result);
    res[1] = cimag(result);
}

void
dhaf(double *mat, int n, double *res)
{
    haf((telem *)mat, n, res);
}


telem
hafnian(telem *mat, int n)
{
    /*
      Add all the terms necessary to calculate the Hafnian of the matrix mat of size n
      It additionally uses open MP to parallelize the summation
    */
    assert(n % 2 == 0);
    sint m = n / 2;
    sint mm = m / 2;

    unsigned long long int pow1 = ((unsigned long long int)pow(2.0, (double)m)) ; //-1;
    unsigned long long int workers = ((unsigned long long int)pow(2.0, (double)mm));
    workers = MIN(workers, pow1);


    unsigned long long int chunksize = pow1;
    unsigned long long int rank = 0;

    telem summand;

    summand = do_chunk(mat, n, rank, chunksize);

    return summand;

}


telem
dhafnian(double *mat, int n)
{
    return hafnian((telem *)mat, n);
}

// Functions for computing Hafnians WITH considering loops (diagonal elements). These functions are named with a suffix "_loops" to differentiate them from previous ones.

telem do_chunk_loops(telem mat[], telem C[], telem D[], int n, unsigned long long int X, unsigned long long int chunksize)
{
    /*
      This function calculates adds parts X to X+chunksize of Cygan and Pilipczuk formula for the
      Hafnian of matrix mat
     */

    telem res = 0.0;
    #pragma omp parallel
    {
        #pragma omp for

        for(unsigned long long int x = X * chunksize; x < (X + 1)*chunksize; x++)
        {

            telem summand = 0.0;
            sint m = n / 2;
            int i, j, k;
            telem factor;
            telem powfactor;


            char dst[m];
            sint pos[n];
            sint sum;
            dec2bin(dst, x, m);
            sum = find2(dst, m, pos);

            double complex B[sum * sum], B_powtrace[sum * sum];
            double complex C1[sum], D1[sum];


            for(i = 0; i < sum; i++)
            {
                for(j = 0; j < sum; j++)
                {
                    B[i * sum + j] = (double complex) mat[pos[i] * n + ((pos[j]) ^ 1)];
                    B_powtrace[i * sum + j] = (double complex) mat[pos[i] * n + ((pos[j]) ^ 1)];
                }
                C1[i] = C[pos[i]];
                D1[i] = D[pos[i]];
            }


            telem traces[m];
            powtrace(B_powtrace, sum, traces, m);
            char cnt;
            sint cntindex;
            cnt = 1;
            cntindex = 0;
            telem comb[2][m + 1];
            for(i = 0; i < n / 2 + 1; i++)
            {
                comb[0][i] = 0.0;
            }
            comb[0][0] = 1.0;

            for(i = 1; i <= n / 2; i++)
            {
                factor = traces[i - 1] / (2 * i);
                // powfactor = 1.0;
                // //the following line is significantly different than in Octave
                // //It basically reassings  the first column of comb to the second and viceversa
                // cnt = -cnt;
                // cntindex = (1 + cnt) / 2;
                // for(j = 0; j < n / 2 + 1; j++)
                // {
                //     comb[1 - cntindex][j] = comb[cntindex][j];
                // }

                // for(j = 1; j <= (n / (2 * i)); j++)
                // {
                //     powfactor = powfactor * factor / j;
                //     for(k = i * j + 1; k <= n / 2 + 1; k++)
                //     {
                //         comb[1 - cntindex][k - 1] = comb[1 - cntindex][k - 1] + comb[cntindex][k - i * j - 1] * powfactor;
                //     }
                // }

                // //  % Combinations from open walks with loops at both ends
                // // This is where the new part goes!

                // factor = 0.0;
                double complex tmpn=0.0;
                for(int i = 0; i < sum; i++){
                    tmpn += C1[i] * D1[i];
                }
                factor += 0.5*tmpn;
                double complex tmp_c1[sum];
                double complex tmp = 0.0;

                for(int i = 0; i < sum; i++)
                {
                    tmp = 0.0;

                    for(int j = 0; j < sum; j++)
                    {
                        tmp += C1[j] * B[j * sum + i];
                    }

                    tmp_c1[i] = tmp;
                }

                for(int i = 0; i < sum; i++)
                    C1[i] = tmp_c1[i];


                powfactor = 1.0;
                //the following line is significantly different than in Octave
                //It basically reassings  the first column of comb to the second and viceversa
                cnt = -cnt;
                cntindex = (1 + cnt) / 2;
                for(j = 0; j < n / 2 + 1; j++)
                {
                    comb[1 - cntindex][j] = comb[cntindex][j];
                }

                for(j = 1; j <= (n / (2 * i)); j++)
                {
                    powfactor = powfactor * factor / j;
                    for(k = i * j + 1; k <= n / 2 + 1; k++)
                    {
                        comb[1 - cntindex][k - 1] = comb[1 - cntindex][k - 1] + comb[cntindex][k - i * j - 1] * powfactor;
                    }
                }

            }


            if(((sum / 2) % 2) == (n / 2 % 2))
            {
                summand = comb[1 - cntindex][n / 2];
            }
            else
            {
                summand = -comb[1 - cntindex][n / 2];
            }
            #pragma omp critical
            res += summand;
        }
    }

    return res;
}


telem hafnian_loops(telem *mat, int n)
{
    /*
      Add all the terms necessary to calculate the Hafnian of the matrix mat of size n
      It additionally uses open MP to parallelize the summation
    */
    assert(n % 2 == 0);
    sint m = n / 2;
    sint mm = m / 2;
    // telem res = 0.0;
    unsigned long long int pow1 = ((unsigned long long int)pow(2.0, (double)m)) ; //-1;
    unsigned long long int workers = ((unsigned long long int)pow(2.0, (double)mm));
    workers = MIN(workers, pow1);


    // telem D_mat[n][n] ;
    telem D[n], C[n];


    unsigned long long int chunksize = pow1;
    unsigned long long int rank = 0;


    for(int i = 0; i < n; i++)
    {
        // for(int j = 0; j < n; j++)
        // {
        //     D_mat[i][j] = 0.0;
        // }
        D[i] = 0.0;
        C[i] = 0.0;
    }
    for(int i = 0; i < n; i++)
    {
        // D_mat[i][i] = mat[i * n + i];
        D[i] = mat[i * n + i];
    }


    for(int i = 0; i < n; i += 2)
    {
        C[i] = D[i + 1];
        C[i + 1] = D[i];
    }

    telem summand;

    summand = do_chunk_loops(mat, C, D, n, rank, chunksize);

    return summand;
}

telem dhafnian_loops(double *mat, int n)
{
    return hafnian_loops((telem *)mat, n);
}

void haf_loops(telem mat[], int n, double res[])
{
    /*
      This is a wrapper for the function hafnian. Instead of returning the Hafnian of the matrix
      mat by value it does by reference with two doubles for the real an imaginary respectively.
    */

    telem result = hafnian_loops(mat, n);
    res[0] = creal(result);
    res[1] = cimag(result);
}

void dhaf_loops(double *mat, int n, double *res)
{
    haf_loops((telem *)mat, n, res);
}
