.DEFAULT_GOAL := main
CFLAGS = -std=c11 -O3 -Wall

main.o: main.c
	gcc $(CFLAGS) -c main.c

misc.o: misc.c misc.h
	gcc $(CFLAGS) -c misc.c

neuron.o: neuron.c neuron.h
	gcc $(CFLAGS) -c neuron.c

layer.o: layer.c layer.h
	gcc $(CFLAGS) -c layer.c

net.o: net.c net.h
	gcc $(CFLAGS) -c net.c

main: main.o misc.o neuron.o layer.o net.o
	gcc $(CFLAGS) -o main main.o misc.o neuron.o layer.o net.o -lm
