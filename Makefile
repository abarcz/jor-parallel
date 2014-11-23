
CXX=g++
CXXFLAGS=-Wall -fopenmp
OBJECTS=main.o

main: $(OBJECTS)
	$(CXX) $(OBJECTS) -fopenmp -o main

main.o: main.cc
	$(CXX) $(CXXFLAGS) -c -o main.o main.cc

clean:
	rm -f main *.o
