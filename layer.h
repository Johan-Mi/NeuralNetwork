#pragma once

#include "neuron.h"

typedef struct {
	unsigned size;
	Neuron* neurons;
} Layer;

Layer makeLayer(const unsigned, const unsigned);
void deleteLayer(Layer*);
