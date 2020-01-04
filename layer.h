#pragma once

#include "neuron.h"

typedef struct {
	uint size;
	Neuron* neurons;
} Layer;

Layer makeLayer(const uint, const uint);
void deleteLayer(Layer*);
Layer copyLayer(const Layer*);
