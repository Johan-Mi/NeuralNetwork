#include "layer.h"
#include "misc.h"

Layer makeLayer(const uint size, const uint prevSize) {
	Layer ret = {size, MKARR(Neuron, size)};
	for(uint i = 0; i < size; i++)
		ret.neurons[i] = makeNeuron(prevSize);
	return ret;
}

void deleteLayer(Layer* layer)  {
	if(layer->neurons) {
		for(uint i = 0; i < layer->size; i++)
			deleteNeuron(&layer->neurons[i]);
		free(layer->neurons);
	}
}
