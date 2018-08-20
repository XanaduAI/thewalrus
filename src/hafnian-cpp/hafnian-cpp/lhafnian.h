#pragma once

#include "stdafx.h"
using namespace std;

typedef complex<double> CplxType;
typedef unsigned char Byte;

void haf(CplxType* mat, int n, double* res);
CplxType hafnian(CplxType* mat, int n);
//void evals (CplxType* z, CplxType* vals, int n);
CplxType hafnian_loops(CplxType* mat, int n);