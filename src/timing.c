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
#include <stdio.h>
#include "lhafnian.h"



int main (){
  int nmax = 10;
  int m, n;
  for (m = 1; m <= nmax; m++)
    {
      n = 2 * m;
      double complex mat[n * n];
      
      int i, j;
      for (i = 0; i < n; i++)
	{
	  for (j = 0; j < n; j++)
	    {
	      mat[n * i + j] = 1.0;
	    }
	}

      double complex hafval = hafnian_loops (mat, n);
  
  
      printf ("%d %lf %lf\n", n, creal (hafval), cimag (hafval) );
      
    }
  return 0;
}
