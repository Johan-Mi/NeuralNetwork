#pragma once

#include "layer.h"

typedef struct NeuralNet {
	uint layerCount;
	Layer* layers;
	T* output;
	uint outputCount;
	T* error;
	T cost;
	struct NeuralNet* delta;
} NeuralNet;

NeuralNet makeNeuralNet(const uint, ...);
void deleteNeuralNet(NeuralNet*);
void think(NeuralNet*, const T*, const T*);
void train(NeuralNet*, uint, const T*, const T*);
