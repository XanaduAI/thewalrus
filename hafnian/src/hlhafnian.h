#ifndef _HAFNIAN_
#define _HAFNIAN_

#include <complex.h>
#include <assert.h>
#include <lapacke.h>
#include <omp.h>

#include <math.h>

//typedef double telem;
typedef unsigned char sint;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void haf(double complex mat[], int n, double res[]);
double hafnian(double complex mat[], int n);
void evals(double complex z[], double vals[], int n);

#endif
