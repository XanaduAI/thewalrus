#ifndef _HAFNIAN_
#define _HAFNIAN_

#include <complex.h>
#include <assert.h>
#include <lapacke.h>
#include <omp.h>

#include <math.h>
#include "hafutils.h"

typedef double complex telem;
typedef unsigned char sint;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void haf(telem mat[], int n, double res[]);
telem hafnian(telem mat[], int n);
void evals(double complex z[], double complex vals[], int n);

#endif
