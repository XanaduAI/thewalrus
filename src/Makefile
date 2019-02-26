# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
EIGEN_INCLUDE_DIR=/usr/include/eigen3

CC=g++
FC=gfortran
FFLAGS=-fopenmp -fPIC -O3 -Wall
CFLAGS=-std=c++11 -O3 -Wall -I/usr/include -I. -I$(EIGEN_INCLUDE_DIR) -fopenmp -march=native
LFLAGS=-L/usr/lib -lm -fopenmp

all: example-f90 example-cpp

example.o: example.cpp hafnian.hpp
	$(CC) $^ $(CFLAGS) -c

example-cpp: example.o
	$(CC) $^ $(LFLAGS) -o $@

example-f90: example.f90 libperm
	$(FC) $< $(FFLAGS) -I. -L. -lperm -o $@

libperm: permanent.F90
	$(FC) $^ $(FFLAGS) -shared -o $@.so

libtor: linpack_q_complex.f90 torontonian.F90
	$(FC) $^ $(FFLAGS) -shared -o $@.so

libhafapprox: linpack_q_complex.f90 hafnian_approx.F90
	$(FC) $^ $(FFLAGS) -shared -o $@.so

clean:
	rm -rf *~ *.out *.o *.so *.pyc *.mod
