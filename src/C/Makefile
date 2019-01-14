# Copyright 2019 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CC=gcc
CFLAGS=-std=c99 -O3 -Wall -I/usr/include -I. -fopenmp
LFLAGS=-L/usr/lib -lm -llapack -llapacke  -lgfortran -lblas -fopenmp
CFLAGS1= -std=c99 -O3 -Wall -fPIC -shared -I/usr/include -fopenmp  -llapacke -I. -o

%.out:%.o lhafnian.o
	$(CC) $^ $(LFLAGS) -o $@

timing.out:timing.o lhafnian.o
	$(CC) $^ $(LFLAGS) -o $@

loopex.out:loopex.o lhafnian.o
	$(CC) $^ $(LFLAGS) -o $@

htiming.out:timing.o hlhafnian.o
	$(CC) $^ $(LFLAGS) -o $@

rtiming.out:timing.o rlhafnian.o
	$(CC) $^ $(LFLAGS) -o $@

library:lhafnian.c
	$(CC) lhafnian.c $(CFLAGS1) lhafnian.so

rlibrary:rlhafnian.c
	$(CC) rlhafnian.c $(CFLAGS1) rlhafnian.so

hlibrary:hlhafnian.c
	$(CC) hlhafnian.c $(CFLAGS1) hlhafnian.so

clean:
	rm -rf *~ *.out *.o *.so *.pyc
