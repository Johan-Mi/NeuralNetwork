#pragma once

typedef struct {
	float activation;
	float bias;
	unsigned connectionCount;
	float* weights;
} Neuron;

Neuron makeNeuron(const unsigned);
void deleteNeuron(Neuron*);
