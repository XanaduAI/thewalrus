# Hafnian C library

Previous version of the Hafnian library were written using C99, and requiring the LAPACKe C bindings to the LAPACK linear algebra library.

As of Hafnian version 0.3.0, the C Hafnian library has been deprecated, and this code ported to C++ (in addition to numerous other Hafnian algorithms and features). This allows us to benefit from:

* Generics, simplifying the amount of code to be maintained.

* Compatibility with the Microsoft Visual C++ 2015 compiler (MSVCC). Unfortunately, MSVCC does not fully support C99, in particular complex number support, leading to the inability to provide Windows compatible Python binaries for Hafnian v0.1.0.

* The C++ header-only Eigen library for linear algebra. This allows compilation on systems without BLAS/LAPACK installed, easing compilation. Furthermore, Eigen can be compiled to use BLAS/LAPACK as a backend if available, allowing use of the highly optimized BLAS/LAPACK routines with minimal additional effort.

To access the new features available in the C++ based Hafnian v0.3.0 and above, please see the main Hafnian documentation.

We leave the C library here for posterity, but it will likely not be updated.

## Installation

To compile the C extension library, you will need the following libraries:

* BLAS
* LAPACKe
* OpenMP

as well as a C99 compliant compiler (i.e. gcc).

On Debian-based systems, the above dependencies can be installed in one go via

```bash
$ sudo apt install liblapacke-dev
```

To compile the library on Linux, a makefile is provided. Simply run

```bash
$ make library
```

and the two libraries `lhafnian.so` (for complex matrices) and `rlhafnian.so` (for real matrices) will be created. Both libraries contain the functions `hafnian_loops(mat, n)` (for the loop hafnian) and `hafnian(mat, n)` (for the standard hafnian), where `mat` is a **flattened** n-by-n array of type `double` or `double complex` depending on the library used.

## Using the C interface

Simply include the required library at the top of your source code:

```c
#include "lhafnian.h"
```

and then you can call the provided hafnian functions directly:

```c
double complex hafval = hafnian_loops(mat, n);
```

To compile, simply provide the path to the corresponding `*lhafnian.so` library, as well as the linking flag. See the file `timing.c` for an example program, and the corresponding Makefile to see how it is compiled.


