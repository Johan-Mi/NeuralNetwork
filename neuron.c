#include "neuron.h"
#include "misc.h"

Neuron makeNeuron(const uint connectionCount) {
	if(connectionCount) {
		Neuron ret = {0, randFloat(), connectionCount, MKARR(T, connectionCount)};
		for(uint i = 0; i < connectionCount; i++)
			ret.weights[i] = randFloat();
		return ret;
	} else {
		return (Neuron){0, 0, 0, NULL};
	}
}

void deleteNeuron(Neuron* neuron) {
	if(neuron->weights)
		free(neuron->weights);
}

Neuron copyNeuron(const Neuron* neuron) {
	Neuron ret;
	ret.connectionCount = neuron->connectionCount;
	ret.weights = MKARR(T, ret.connectionCount);
	ret.bias = 0;
	for(uint i = 0; i < ret.connectionCount; i++) {
		ret.weights[i] = 0;
	}

	return ret;
}
