#pragma once

#include "misc.h"

typedef struct {
	T activation;
	T bias;
	uint connectionCount;
	T* weights;
	T preSigmoid;
} Neuron;

Neuron makeNeuron(const uint);
void deleteNeuron(Neuron*);
