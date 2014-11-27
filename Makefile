
CC=gcc
CCFLAGS=-Wall -std=c99 -fopenmp -O3
OBJECTS=main.o amatrix.o

main: $(OBJECTS)
	$(CC) $(OBJECTS) -fopenmp -lm -o main

main.o: main.c amatrix.h
	$(CC) $(CCFLAGS) -c -o main.o main.c

amatrix.o: amatrix.c amatrix.h
	$(CC) $(CCFLAGS) -c -o amatrix.o amatrix.c

main.s: main.c
	$(CC) $(CCFLAGS) -c -g -Wa,-aslh main.c > main.s

clean:
	rm -f main *.o main.s
