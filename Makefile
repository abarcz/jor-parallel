
CC=gcc
CCFLAGS=-Wall -std=c99 -fopenmp -O3
OBJECTS=main.o amatrix.o deviceid.o

NVCC=nvcc
NVCC_CCFLAGS=-ccbin g++ -I../../common/inc -m64 -gencode arch=compute_20,code=sm_20

main: $(OBJECTS)
	$(NVCC) $(OBJECTS) -lgomp -lm -o main

main.o: main.c amatrix.h deviceid.h common.h
	$(NVCC) $(NVCC_CCFLAGS) -Xcompiler "$(CCFLAGS)" -o $@ -c $<

amatrix.o: amatrix.c amatrix.h common.h
	$(NVCC) $(NVCC_CCFLAGS) -Xcompiler "$(CCFLAGS)" -o $@ -c $<

deviceid.o: deviceid.cu deviceid.h common.h
	$(NVCC) $(NVCC_CCFLAGS) -o $@ -c $<

main.s: main.c
	$(CC) $(CCFLAGS) -c -g -Wa,-aslh main.c > main.s

clean:
	rm -f main *.o main.s
