#pragma once

#include "layer.h"

typedef struct {
	uint layerCount;
	Layer* layers;
	T* output;
	uint outputCount;
	T* error;
	T cost;
} NeuralNet;

NeuralNet makeNeuralNet(const uint, ...);
void deleteNeuralNet(NeuralNet*);
void think(NeuralNet*, const T*, const T*);
