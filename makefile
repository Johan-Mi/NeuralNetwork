CFLAGS = -std=c++20 -O3 -Wall -march=native
CC = g++

main: main.o
	${CC} ${CFLAGS} -o main main.o -lm -lfmt

main.o: main.cpp net.hpp misc.hpp
	${CC} ${CFLAGS} -c main.cpp

clean:
	rm -f *.o main
