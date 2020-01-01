#pragma once

#include "layer.h"

typedef struct {
	unsigned layerCount;
	Layer* layers;
	float* output;
	unsigned outputCount;
	float* error;
	float cost;
} NeuralNet;

NeuralNet makeNeuralNet(const unsigned, ...);
void deleteNeuralNet(NeuralNet*);
void think(NeuralNet*, const float*, const float*);
