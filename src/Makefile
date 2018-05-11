########################
# Makefile for lapacke #
# Nicolas Quesada      #
########################


CC=gcc
CFLAGS1=-std=c99 -O3 -Wall -I/usr/include -I. -fopenmp
LFLAGS=-L/usr/lib -lm -llapack -llapacke  -lgfortran -lblas -fopenmp
CFLAGS= -std=c99 -O3 -Wall -fPIC -shared -I/usr/include -fopenmp  -llapacke -I. -o


%.out:%.o lhafnian.o
	$(CC) $^ $(LFLAGS) -o $@

timing.out:timing.o lhafnian.o 
	$(CC) $^ $(LFLAGS) -o $@

rtest.out:rtest.o rlhafnian.o
	$(CC) $^ $(LFLAGS) -o $@

library:lhafnian.c
	$(CC) lhafnian.c $(CFLAGS) lhafnian.so

rlibrary:rlhafnian.c
	$(CC) rlhafnian.c $(CFLAGS) rlhafnian.so

hlibrary:hlhafnian.c
	$(CC) hlhafnian.c $(CFLAGS) hlhafnian.so

clean:
	rm -rf *~ *.out *.o *.so *.pyc
