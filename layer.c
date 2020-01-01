#include "layer.h"
#include "misc.h"

Layer makeLayer(const unsigned size, const unsigned prevSize) {
	Layer ret = {size, MKARR(Neuron, size)};
	for(unsigned i = 0; i < size; i++)
		ret.neurons[i] = makeNeuron(prevSize);
	return ret;
}

void deleteLayer(Layer* layer)  {
	if(layer->neurons) {
		for(unsigned i = 0; i < layer->size; i++)
			deleteNeuron(&layer->neurons[i]);
		free(layer->neurons);
	}
}
