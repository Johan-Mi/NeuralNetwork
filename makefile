.DEFAULT_GOAL := main
CFLAGS = -std=c++17 -O3 -Wall

main.o: main.cpp net.hpp layer.hpp neuron.hpp
	g++ $(CFLAGS) -c main.cpp

misc.o: misc.cpp misc.hpp
	g++ $(CFLAGS) -c misc.cpp

main: main.o misc.o
	g++ $(CFLAGS) -o main main.o misc.o -lm

clean:
	rm -f *.o main
