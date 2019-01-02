# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
EIGEN_INCLUDE_DIR=/home/josh/.local/include/

CC=g++
CFLAGS=-std=c++11 -O3 -Wall -I/usr/include -I. -I$(EIGEN_INCLUDE_DIR) -fopenmp
LFLAGS=-L/usr/lib -lm -fopenmp

all: timing

timing.o:timing.cpp hafnian.hpp
	$(CC) timing.cpp $(CFLAGS) -c

timing: timing.o
	$(CC) $^ $(LFLAGS) -o $@

clean:
	rm -rf *~ *.out *.o *.so *.pyc
