########################
# Makefile for lapacke #
# Nicolas Quesada      #
########################


CC=g++
CFLAGS=-std=c++11 -O3 -Wall -I/usr/include -I. -fopenmp
LFLAGS=-L/usr/lib -lm -fopenmp


timing.o:timing.cpp hafnian.hpp
	$(CC) timing.cpp $(CFLAGS) -c

timing: timing.o
	$(CC) $^ $(LFLAGS) -o $@


clean:
	rm -rf *~ *.out *.o *.so *.pyc
