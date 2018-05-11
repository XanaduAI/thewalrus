########################
# Makefile for lapacke #
# Nicolas Quesada      #
########################


CC=gcc
#CFLAGS=-std=c99 -O3 -Wall -I/usr/include -I. -fopenmp
LFLAGS=-L/usr/lib -lm -llapack -llapacke  -lgfortran -lblas -fopenmp
CFLAGS1= -std=c99 -O3 -Wall -fPIC -shared -I/usr/include -fopenmp  -llapacke -I. -o


%.out:%.o 
	$(CC) $^ $(LFLAGS) -o $@

htiming.out:htiming.o hlhafnian.o
	$(CC) $^ $(LFLAGS) -o $@

timing.out:timing.o lhafnian.o hafutils.o
	$(CC) $^ $(LFLAGS) -o $@

rtiming.out:rtiming.o rlhafnian.o
	$(CC) $^ $(LFLAGS) -o $@

library:lhafnian.c hafutils.c
	$(CC) lhafnian.c $(CFLAGS1) lhafnian.so

rlibrary:rlhafnian.c
	$(CC) rlhafnian.c $(CFLAGS1) rlhafnian.so 

hlibrary:hlhafnian.c
	$(CC) hlhafnian.c $(CFLAGS1) hlhafnian.so 

clean:
	rm -rf *~ *.out *.o *.so *.pyc
