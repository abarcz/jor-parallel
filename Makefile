
CC=gcc
CCFLAGS=-Wall -std=c99 -fopenmp -O3
OBJECTS=main.o

main: $(OBJECTS)
	$(CC) $(OBJECTS) -fopenmp -lm -o main

main.o: main.c
	$(CC) $(CCFLAGS) -c -o main.o main.c

main.s: main.c
	$(CC) $(CCFLAGS) -lm -g -Wa,-aslh main.c > main.s

clean:
	rm -f main *.o main.s
