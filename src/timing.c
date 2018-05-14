#include <stdio.h>
#include "lhafnian.h"
#include <sys/time.h>


int
main ()
{
  int nmax = 20;
  int m, n;
  for (m = 1; m <= nmax; m++)
    {
      n = 2 * m;
      telem mat[n * n];

      int i, j;
      for (i = 0; i < n; i++)
	{
	  for (j = 0; j < n; j++)
	    {
	      mat[n * i + j] = 1.0;
	    }
	}

      struct timeval tv1, tv2;
      gettimeofday (&tv1, NULL);
      telem haf = hafnian (mat, n);
      gettimeofday (&tv2, NULL);

      double time =
	(double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
	(double) (tv2.tv_sec - tv1.tv_sec);

      printf ("%d %f %f\n", n, time, creal (haf));
    }


  for (m = 1; m <= nmax; m++)
    {
      n = 2 * m;
      telem mat[n * n];

      int i, j;
      for (i = 0; i < n; i++)
	{
	  for (j = 0; j < n; j++)
	    {
	      mat[n * i + j] = 0.0 * (1.0 / (1 + i + j)) + 1.0;
	    }
	}
      mat[n * 1 + 0] = 0.0;
      mat[n * 0 + 1] = 0.0;


      struct timeval tv1, tv2;
      gettimeofday (&tv1, NULL);
      telem haf = hafnian (mat, n);
      gettimeofday (&tv2, NULL);

      double time =
	(double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
	(double) (tv2.tv_sec - tv1.tv_sec);

      printf ("%d %f %f\n", n, time, creal (haf));
    }

  return 0;
}
