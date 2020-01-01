#include "neuron.h"
#include "misc.h"

Neuron makeNeuron(const unsigned connectionCount) {
	if(connectionCount) {
		Neuron ret = {0, randFloat(), connectionCount, MKARR(float, connectionCount)};
		for(unsigned i = 0; i < connectionCount; i++)
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
